"""Official 50,000-image V1 accuracy and every-logit CPU/NPU numerical validation."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from common import ROOT,PROFILE,own_path,write_json,sha256


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['prepare','cpu','npu','both'],default='both')
    p.add_argument('--archives',type=Path,default=ROOT/'data/imagenet_official')
    p.add_argument('--index',type=Path,default=ROOT/'data/imagenet_val_index')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--devices',default='0');p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--worker',action='store_true')
    return p


def launch(a):
    import fcntl
    archives=own_path(a.archives,exists=True);index=own_path(a.index);output=own_path(a.output_dir)
    if a.batch_size<1 or a.workers<0:raise RuntimeError('Invalid evaluation batch/workers.')
    spec=json.loads((ROOT/'images.lock.json').read_text())['images']['runtime']
    actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}|{{.Architecture}}',spec['tag']],text=True).strip()
    if actual!=spec['image_id']+'|arm64':raise RuntimeError('Runtime image ID or architecture differs.')
    output.mkdir(parents=True,exist_ok=True)
    phases=['cpu','npu'] if a.backend=='both' else [a.backend]
    with (ROOT/'.imagenet_validation.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in phases:
            command=['docker','run','--rm','--network','none','--memory','24g','--cpus','16','--shm-size','4g',
                '-v',str(ROOT)+':'+str(ROOT)+':rw','-w',str(ROOT)]
            for k,v in {'OMP_NUM_THREADS':'8' if phase=='cpu' else '2','OPENBLAS_NUM_THREADS':'2',
                'PYTHONUNBUFFERED':'1','TORCH_DEVICE_BACKEND_AUTOLOAD':'0' if phase!='npu' else '1'}.items():command+=['-e',k+'='+v]
            if phase=='npu':
                if not a.devices.isdigit():raise RuntimeError('This accuracy command selects exactly one logical NPU.')
                subprocess.run([sys.executable,str(ROOT/'tools/check_idle_cards.py'),a.devices],check=True)
                for path in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi']:
                    if Path(path).exists():command+=['-v',path+':'+path+':ro']
                for path in ['/dev/davinci'+a.devices,'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
                    command+=['--device',path]
                command+=['-e','ASCEND_RT_VISIBLE_DEVICES=0']
                stamp=ROOT/'test/output/last_container_exit.json'
                if stamp.exists():time.sleep(max(0,8-(time.time()-json.loads(stamp.read_text())['finished_at'])))
            command+=['--entrypoint',PROFILE['python'],spec['tag'],'tools/validate_imagenet.py','--worker',
                '--backend',phase,'--archives',str(archives),'--index',str(index),'--output_dir',str(output),
                '--batch_size',str(a.batch_size),'--workers',str(a.workers)]
            with (output/(phase+'_console.log')).open('wb') as stream:
                code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
            if phase=='npu':write_json(ROOT/'test/output/last_container_exit.json',{'finished_at':time.time(),'exit_code':code})
            if code:raise RuntimeError(phase+' evaluation failed; see '+str(output/(phase+'_console.log')))


def work(a):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from imagenet_archive import SPEC,prepare_validation,ValidationTarDataset,verify_archive
    if a.backend=='prepare':
        print(json.dumps(prepare_validation(a.archives,a.index),indent=2));return
    if not (a.index/'metadata.json').exists():prepare_validation(a.archives,a.index)
    output=own_path(a.output_dir,exists=True);result_file=output/(a.backend+'_result.json')
    if result_file.exists():raise RuntimeError('Use a new output directory; previous accuracy results are preserved.')
    dataset=ValidationTarDataset(a.index)
    # Recheck the full official archive in each independent evaluation process.
    archive_receipt=verify_archive(dataset.archive)
    locked=next(x for x in dataset.metadata['archive_receipts'] if x['name']==dataset.archive.name)
    if archive_receipt!=locked:raise RuntimeError('Official archive changed since indexing.')
    weight=own_path(a.archives/SPEC['weights']['filename'],exists=True)
    if sha256(weight)!=SPEC['weights']['sha256']:raise RuntimeError('Official ModelZoo V1 weight digest differs.')
    torch.set_num_threads(8 if a.backend=='cpu' else 2);torch.use_deterministic_algorithms(True)
    device=torch.device('cpu')
    if a.backend=='npu':
        import torch_npu
        from npu_precision import configure_fp32
        precision=configure_fp32(torch_npu)
        torch_npu.npu.set_device(0);device=torch.device('npu:0')
    import model_task as task
    state=torch.load(weight,map_location='cpu',weights_only=True)
    model=task.create_model({});model.load_state_dict(state,strict=True);model.eval();model.to(device)
    reference=None
    if a.backend=='npu':
        cpu_result=json.loads((output/'cpu_result.json').read_text())
        if cpu_result['result']!='pass_official_full_validation' or cpu_result['weight_sha256']!=sha256(weight):
            raise RuntimeError('A passing independent CPU reference is required first.')
        if cpu_result['index_metadata_sha256']!=sha256(a.index/'metadata.json'):
            raise RuntimeError('CPU and NPU validation index metadata differ.')
        if sha256(output/'cpu_logits.npy')!=cpu_result['logits_sha256']:raise RuntimeError('CPU logit reference changed.')
        reference=np.load(output/'cpu_logits.npy',mmap_mode='r',allow_pickle=False)
    else:
        reference=np.lib.format.open_memmap(output/'cpu_logits.partial.npy',mode='w+',dtype=np.float32,shape=(50000,1000))
    options={'batch_size':a.batch_size,'shuffle':False,'num_workers':a.workers,'drop_last':False}
    if a.workers:options.update(multiprocessing_context='spawn',persistent_workers=True)
    loader=DataLoader(dataset,**options)
    counts=[0,0];seen=0;max_error=0.0;violations=0;start=time.monotonic();architecture_checked=False
    with torch.no_grad():
        for images,labels,indices in loader:
            if not torch.equal(indices,torch.arange(seen,seen+len(images))):raise RuntimeError('Validation order/coverage differs.')
            if not architecture_checked and a.backend=='cpu':
                from torchvision.models import mobilenet_v2
                official=mobilenet_v2(weights=None).eval();official.load_state_dict(state,strict=True)
                torch.testing.assert_close(model(images[:2]),official(images[:2]),rtol=1e-5,atol=1e-6)
                architecture_checked=True;del official
            logits=model(images.to(device)).float().cpu().numpy()
            if not np.isfinite(logits).all():raise RuntimeError('Nonfinite full-validation logits.')
            target=labels.numpy();top=np.argsort(logits,axis=1)[:,-5:]
            counts[0]+=int((logits.argmax(1)==target).sum());counts[1]+=int((top==target[:,None]).any(1).sum())
            if a.backend=='cpu':reference[seen:seen+len(images)]=logits
            else:
                expected=reference[seen:seen+len(images)];delta=np.abs(logits-expected)
                max_error=max(max_error,float(delta.max()))
                gate=SPEC['engineering_acceptance'];violations+=int((delta>(gate['all_logit_atol']+gate['all_logit_rtol']*np.abs(expected))).sum())
            seen+=len(images)
            if seen%6400==0 or seen==50000:
                progress={'backend':a.backend,'images':seen,'expected':50000,'seconds':time.monotonic()-start,
                    'top1_percent_so_far':100*counts[0]/seen,'top5_percent_so_far':100*counts[1]/seen}
                write_json(output/(a.backend+'_progress.json'),progress);print(json.dumps(progress),flush=True)
    if seen!=50000:raise RuntimeError('Full validation requires all 50,000 images, exactly once.')
    gate=SPEC['engineering_acceptance'];scores=[100*x/seen for x in counts]
    if a.backend=='cpu':
        reference.flush();del reference;(output/'cpu_logits.partial.npy').replace(output/'cpu_logits.npy')
        delta=[abs(scores[i]-SPEC['weights'][key]) for i,key in enumerate(['reference_top1_percent','reference_top5_percent'])]
    else:delta=[abs(scores[i]-cpu_result[key]) for i,key in enumerate(['top1_percent','top5_percent'])]
    passed=violations==0 and max(delta)<=gate['top1_top5_delta_percentage_points']
    result={'model':PROFILE['model'],'backend':a.backend,'result':'pass_official_full_validation' if passed else 'fail',
        'data_scope':dataset.metadata['scope'],'images_evaluated':seen,'classes':1000,
        'top1_correct':counts[0],'top5_correct':counts[1],'top1_percent':scores[0],'top5_percent':scores[1],
        'accuracy_deltas_percentage_points':delta,'comparison':'published V1 reference' if a.backend=='cpu' else 'independent complete CPU evaluation',
        'weight_sha256':sha256(weight),'index_metadata_sha256':sha256(a.index/'metadata.json'),
        'source_architecture_control_passed':architecture_checked if a.backend=='cpu' else cpu_result['source_architecture_control_passed'],
        'every_logit_compared':a.backend=='npu','logits_compared':50000000 if a.backend=='npu' else 0,
        'logit_max_absolute_error':max_error if a.backend=='npu' else None,'logits_outside_tolerance':violations,
        'engineering_acceptance':gate,'total_evaluation_seconds':time.monotonic()-start,
        'training_from_scratch_claimed':False,'official_pretrained_weights_used':True}
    result['npu_precision']=precision if a.backend=='npu' else None
    if a.backend=='cpu':result['logits_sha256']=sha256(output/'cpu_logits.npy')
    write_json(result_file,result);print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('Full ImageNet accuracy/numerical gate failed; all measurements preserved.')


if __name__=='__main__':
    args=parser().parse_args()
    if args.worker:work(args)
    else:launch(args)
