"""Independent real-data gradient/optimizer, throughput and sustained-training validation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, PROFILE, own_path, sha256, write_json, tensor_tree_cpu, finite_tree, parameter_sha


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',choices=['gradient','benchmark','stability'],required=True)
    p.add_argument('--mode',choices=['functional','full'],default='functional')
    p.add_argument('--devices',default=PROFILE['host_devices'])
    p.add_argument('--nproc',type=int,default=1)
    p.add_argument('--data_path',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--duration_seconds',type=int,default=1800)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--worker',action='store_true')
    return p


def launch(a):
    import fcntl
    if a.mode=='full' and (a.check!='gradient' or a.checkpoint is None):
        raise RuntimeError('Full-data control requires --check gradient and an explicit selected full checkpoint; complete curriculum supplies long-run evidence.')
    if a.check=='gradient' and a.nproc!=1:raise RuntimeError('CPU/NPU gradient control uses one NPU.')
    if a.nproc<1 or a.duration_seconds<1 or a.warmup<1 or a.iterations<2:raise RuntimeError('Invalid validation parameters.')
    data=own_path(a.data_path or ROOT/PROFILE['data'][a.mode],exists=True)
    checkpoint=own_path(a.checkpoint or ROOT/'test/output/delivery_verified/checkpoint.pt',exists=True)
    output=own_path(a.output_dir)
    ids=a.devices.split(',')[:a.nproc]
    if len(ids)!=a.nproc or any(not x.isdigit() for x in ids) or len({int(x)//2 for x in ids})!=a.nproc:
        raise RuntimeError('Select one logical ID from each distinct physical NPU.')
    spec=json.loads((ROOT/'images.lock.json').read_text())['images']['runtime']
    probe=subprocess.run(['docker','image','inspect','--format','{{.Id}}|{{.Architecture}}',spec['tag']],capture_output=True,text=True,check=True)
    if probe.stdout.strip()!=spec['image_id']+'|arm64':raise RuntimeError('The validated runtime image differs.')
    with (ROOT/'.extended_validation.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(ROOT/'tools/check_idle_cards.py'),','.join(ids)],check=True)
        output.mkdir(parents=True,exist_ok=False)
        command=['docker','run','--rm','--network','host','--memory','96g','--cpus','16','--pids-limit','1024','--shm-size','8g',
                 '-v',str(ROOT)+':'+str(ROOT)+':rw','-w',str(ROOT)]
        for path in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi','/usr/local/bin/npu-smi']:
            if Path(path).exists():command+=['-v',path+':'+path+':ro']
        for path in ['/dev/davinci'+x for x in ids]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
            if not Path(path).is_char_device():raise RuntimeError('Missing NPU character device: '+path)
            command+=['--device',path]
        for name,value in {'ASCEND_RT_VISIBLE_DEVICES':','.join(str(x) for x in range(a.nproc)),
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','HCCL_CONNECT_TIMEOUT':'180'}.items():
            command+=['-e',name+'='+value]
        command+=['--entrypoint',PROFILE['python'],spec['tag'],'-m','torch.distributed.run','--standalone',
            '--nproc_per_node='+str(a.nproc),'tools/validate_extended.py','--worker','--check',a.check,
            '--mode',a.mode,
            '--data_path',str(data),'--checkpoint',str(checkpoint),'--output_dir',str(output),
            '--duration_seconds',str(a.duration_seconds),'--warmup',str(a.warmup),'--iterations',str(a.iterations)]
        state={'model':PROFILE['model'],'check':a.check,'state':'running','host_logical_device_ids':ids,'physical_device_ids':[int(x)//2 for x in ids],'image_id':spec['image_id']}
        write_json(output/'execution.json',state)
        stamp=ROOT/'test/output/last_container_exit.json'
        if stamp.is_file():
            remaining=8-(time.time()-json.loads(stamp.read_text())['finished_at'])
            if remaining>0:time.sleep(remaining)
        with (output/'console.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            for block in iter(lambda:process.stdout.read1(65536),b''):
                log.write(block);log.flush();sys.stdout.buffer.write(block);sys.stdout.buffer.flush()
            code=process.wait()
        write_json(stamp,{'finished_at':time.time(),'exit_code':code})
        state.update(state='completed' if code==0 else 'failed',exit_code=code)
        write_json(output/'execution.json',state)
        return code


def work(a):
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader,DistributedSampler
    import model_task as task
    rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.set_device(local_rank);device=torch.device('npu',local_rank)
    if world>1:dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=600))
    output=own_path(a.output_dir,exists=True);data=own_path(a.data_path or ROOT/PROFILE['data'][a.mode],exists=True)
    metadata=task.check_data(data,a.mode)
    checkpoint=own_path(a.checkpoint or ROOT/'test/output/delivery_verified/checkpoint.pt',exists=True)
    original=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if original['model']!=PROFILE['model'] or original['dataset_digest']!=sha256(data/'metadata.json'):
        raise RuntimeError('The source checkpoint model or functional data digest differs.')
    model=task.create_model(metadata).to(device);model.load_state_dict(original['state_dict'],strict=True)
    if any(p.device.type!='npu' for p in model.parameters()):raise RuntimeError('Parameters must all be on NPU.')
    train,validation=task.datasets(data,metadata)
    batch_size=512 if a.mode=='full' else PROFILE['training']['functional']['batch_size']
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=42,drop_last=True)
    loader=DataLoader(train,batch_size=batch_size,sampler=sampler,drop_last=True,num_workers=0,collate_fn=task.collate)
    if not len(loader):raise RuntimeError('No real-data validation batches.')
    base={'model':PROFILE['model'],'check':a.check,'mode':a.mode,'data_scope':metadata['scope'],
          'source_checkpoint_sha256':sha256(checkpoint),'world_size':world,'per_rank_batch_size':batch_size,
          'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'formal_accuracy_claimed':False}
    if a.check=='gradient':
        batch=next(iter(loader));model.eval()
        cpu=task.create_model(metadata).cpu();cpu.load_state_dict(original['state_dict'],strict=True);cpu.eval()
        if a.mode=='full':
            lr=original['settings']['lr']
            npu_optimizers=[torch.optim.Adam(model.encoder.parameters(),lr=lr),torch.optim.Adam(model.decoder.parameters(),lr=lr)]
            cpu_optimizers=[torch.optim.Adam(cpu.encoder.parameters(),lr=lr),torch.optim.Adam(cpu.decoder.parameters(),lr=lr)]
        else:
            npu_optimizers=[task.create_optimizer(model,original['settings'])]
            cpu_optimizers=[task.create_optimizer(cpu,original['settings'])]
        loss,_=task.loss(model,batch,device);loss.backward()
        expected,_=task.loss(cpu,batch,torch.device('cpu'));expected.backward()
        gradients={name:p.grad.detach().cpu() if p.grad is not None else None for name,p in model.named_parameters()}
        rows=[];difference_norm=reference_norm=0.0;maximum=0.0;matched=True;active=0
        for name,p in cpu.named_parameters():
            actual=gradients[name];reference=p.grad
            if actual is None or reference is None:
                if actual is not None or reference is not None:matched=False
                rows.append({'parameter':name,'active':False,'both_inactive':actual is None and reference is None});continue
            active+=1
            delta=(actual-reference).double();error=float(delta.abs().max());refmax=float(reference.abs().max())
            valid=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()) and error<=0.001+0.01*refmax
            matched=matched and valid;maximum=max(maximum,error)
            difference_norm+=float(delta.square().sum());reference_norm+=float(reference.double().square().sum())
            rows.append({'parameter':name,'active':True,'max_absolute_error':error,'reference_max':refmax,'pass':valid})
        relative=(difference_norm/max(reference_norm,1e-24))**0.5
        for optimizer in npu_optimizers:optimizer.step()
        for optimizer in cpu_optimizers:optimizer.step()
        step_error=0.0;step_pass=True
        for (name,p),(other,q) in zip(model.named_parameters(),cpu.named_parameters()):
            assert name==other
            actual=p.detach().cpu();reference=q.detach()
            step_error=max(step_error,float((actual-reference).abs().max()))
            step_pass=step_pass and torch.allclose(actual,reference,rtol=0.003,atol=0.0003)
        loss_pass=torch.allclose(loss.detach().cpu(),expected.detach(),rtol=0.003,atol=0.003)
        passed=matched and active>0 and relative<=0.01 and step_pass and bool(loss_pass)
        result={**base,'result':'pass' if passed else 'fail','control_mode':'eval with autograd; identical real training batch and stored weights',
            'optimizer_state_source':'fresh identical optimizers for a one-step numerical control; separate continuation tests restore saved optimizer state',
            'optimizer_structure':'original independent encoder Adam and decoder Adam' if a.mode=='full' else 'functional Adam',
            'cpu_loss':float(expected.detach()),'npu_loss':float(loss.detach().cpu()),'loss_close':bool(loss_pass),
            'active_gradient_tensors':active,'all_active_and_inactive_parameters_match':matched,
            'gradient_max_absolute_error':maximum,'gradient_relative_l2_error':relative,
            'gradient_relative_l2_limit':0.01,'gradient_per_tensor_limit':'0.001 + 0.01 * reference_max',
            'optimizer_step_parameter_max_error':step_error,'optimizer_step_parameters_close':bool(step_pass),
            'optimizer_parameter_rtol':0.003,'optimizer_parameter_atol':0.0003,'parameters':rows}
        write_json(output/'gradient_control.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='parameters'},indent=2),flush=True)
        if not passed:raise RuntimeError('Real-data CPU/NPU gradient or optimizer-step control failed.')
        return
    optimizer=task.create_optimizer(model,original['settings'])
    if 'optimizer' not in original or 'param_groups' not in original['optimizer']:
        raise RuntimeError('Stability/benchmark starts from a model-and-optimizer delivery checkpoint.')
    optimizer.load_state_dict(original['optimizer'])
    initial=parameter_sha(model)
    wrapped=DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=PROFILE.get('find_unused_parameters',False)) if world>1 else model
    model.train();epoch=steps=0;sampler.set_epoch(epoch);iterator=iter(loader)
    def update():
        nonlocal epoch,iterator
        try:batch=next(iterator)
        except StopIteration:
            epoch+=1;sampler.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss,detail=task.loss(wrapped,batch,device)
        if loss.device.type!='npu' or not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite / non-NPU sustained-training loss.')
        loss.backward()
        if a.check=='stability' and steps%10==0:
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(g.device.type!='npu' or not bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError('Missing / nonfinite / non-NPU sustained-training gradients.')
        if PROFILE.get('clip_grad_norm'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),PROFILE['clip_grad_norm'],foreach=False,error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach().cpu())
    if a.check=='benchmark':
        for _ in range(a.warmup):update();steps+=1
        torch_npu.npu.synchronize()
        if world>1:dist.barrier()
        if hasattr(torch_npu.npu,'reset_peak_memory_stats'):torch_npu.npu.reset_peak_memory_stats()
        latencies=[];begin=time.monotonic()
        for _ in range(a.iterations):
            start=time.monotonic();update();steps+=1;torch_npu.npu.synchronize();latencies.append(time.monotonic()-start)
        elapsed=time.monotonic()-begin
        durations=torch.tensor(latencies,device=device)
        wall=torch.tensor([elapsed],device=device)
        if world>1:dist.all_reduce(durations,op=dist.ReduceOp.MAX);dist.all_reduce(wall,op=dist.ReduceOp.MAX)
        result={**base,'result':'pass_measured_real_training_throughput','warmup_updates':a.warmup,
            'timed_updates':a.iterations,'timed_seconds':float(wall.cpu()[0]),
            'global_samples_per_second':a.iterations*batch_size*world/float(wall.cpu()[0]),
            'latency_p50_seconds':float(np.percentile(durations.cpu().numpy(),50)),
            'latency_p95_seconds':float(np.percentile(durations.cpu().numpy(),95)),
            'peak_allocated_bytes_local_rank':torch_npu.npu.max_memory_allocated() if hasattr(torch_npu.npu,'max_memory_allocated') else None,
            'includes':'real data loading, host-to-NPU, forward, backward, optimizer; synchronized each update',
            'scaling_method':'constant per-rank batch; compare separately measured 1p and multiprocess runs',
            'performance_target_provided':False}
        write_json(output/('benchmark_rank_'+str(rank)+'.json'),result)
        if rank==0:print(json.dumps(result,indent=2),flush=True)
    else:
        begin=time.monotonic();last=0.0
        while True:
            last=update();steps+=1
            if steps%10==0:
                elapsed=time.monotonic()-begin
                stop=torch.tensor([int(elapsed>=a.duration_seconds) if rank==0 else 0],device=device)
                if world>1:dist.broadcast(stop,src=0)
                if rank==0 and steps%100==0:print(json.dumps({'updates':steps,'seconds':elapsed,'loss':last}),flush=True)
                if bool(stop.cpu()[0]):break
        elapsed=time.monotonic()-begin;final=parameter_sha(model)
        if final==initial or not finite_tree(model.state_dict()) or not finite_tree(optimizer.state_dict()):
            raise RuntimeError('Sustained-training parameters did not update or state is nonfinite.')
        if world>1:
            code=torch.tensor(list(bytes.fromhex(final)),device=device,dtype=torch.int32)
            codes=[torch.empty_like(code) for _ in range(world)];dist.all_gather(codes,code)
            if any(not torch.equal(code,value) for value in codes):raise RuntimeError('Sustained-training rank parameter SHA differs.')
        result={**base,'result':'pass_sustained_real_data_training','requested_seconds':a.duration_seconds,
            'actual_training_seconds':elapsed,'updates_this_run':steps,'last_loss':last,
            'state_finite':True,'optimizer_finite':True,'parameters_updated':True,'parameter_sha256':final,
            'all_rank_parameter_hashes_match':True,'batchnorm_buffers_not_claimed_identical':True}
        write_json(output/('stability_rank_'+str(rank)+'.json'),result)
        if rank==0:
            state={'model':PROFILE['model'],'state_dict':tensor_tree_cpu(model.state_dict()),'optimizer':tensor_tree_cpu(optimizer.state_dict()),
                'step':original['step']+steps,'epoch':epoch,'world_size':world,'dataset_digest':sha256(data/'metadata.json'),
                'metadata':metadata,'settings':original['settings']}
            torch.save(state,output/'checkpoint.pt')
            model.eval();sample=next(iter(DataLoader(validation,batch_size=batch_size,num_workers=0,collate_fn=task.collate)))
            with torch.no_grad():reference=task.predict(model,sample,device).cpu()
            torch.save({'batch':tensor_tree_cpu(sample),'prediction':reference},output/'reload_reference.pt')
            write_json(output/'train_result.json',{'model':PROFILE['model'],'mode':'functional','result':'pass_sustained_real_data_training',
                'checkpoint_sha256':sha256(output/'checkpoint.pt'),'optimizer_steps':state['step'],'world_size':world,
                'data_path':data.relative_to(ROOT).as_posix(),'data_scope':metadata['scope'],'settings':original['settings'],
                'formal_accuracy_claimed':False,'resume_from':str(checkpoint)})
            print(json.dumps(result,indent=2),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    args=parser().parse_args()
    if args.worker:work(args)
    else:raise SystemExit(launch(args))
