"""One command for the original 65-frame mini planning/collision numerical control."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from common import ROOT,PROFILE,own_path,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--devices',default='0')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'test/output/delivery_verified/checkpoint.pt')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--data_path',type=Path,default=ROOT/'data/nuscenes')
    p.add_argument('--hf32',choices=['runtime_default','off'],default='off')
    a=p.parse_args();checkpoint=own_path(a.checkpoint,exists=True);output=own_path(a.output_dir);data=own_path(a.data_path,exists=True)
    if not a.devices.isdigit():raise RuntimeError('Select one logical NPU ID.')
    if output.exists():raise RuntimeError('Keep the previous result and select a new output directory.')
    spec=json.loads((ROOT/'images.lock.json').read_text())['images']['runtime']
    actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}|{{.Architecture}}',spec['tag']],text=True).strip()
    if actual!=spec['image_id']+'|arm64':raise RuntimeError('Pinned runtime image differs.')
    import fcntl
    with (ROOT/'.model_operation.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(ROOT/'tools/check_idle_cards.py'),a.devices],check=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        command=['docker','run','--rm','--network','none','--memory','32g','--cpus','8','--shm-size','4g',
            '-v',str(ROOT)+':'+str(ROOT)+':rw','-w',str(ROOT),'-e','ASCEND_RT_VISIBLE_DEVICES=0',
            '-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-e','PYTHONUNBUFFERED=1']
        for path in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/usr/local/dcmi']:
            if Path(path).exists():command+=['-v',path+':'+path+':ro']
        for path in ['/dev/davinci'+a.devices,'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
            if not Path(path).is_char_device():raise RuntimeError('Missing device '+path)
            command+=['--device',path]
        command+=['--entrypoint',PROFILE['python'],spec['tag'],'tools/validate_planning.py',
            '--data_path',str(data),'--checkpoint',str(checkpoint),'--output_dir',str(output),'--hf32',a.hf32]
        stamp=ROOT/'test/output/last_container_exit.json'
        if stamp.exists():time.sleep(max(0,8-(time.time()-json.loads(stamp.read_text())['finished_at'])))
        log=output.parent/(output.name+'_console.log')
        with log.open('xb') as stream:code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
        write_json(stamp,{'finished_at':time.time(),'exit_code':code})
        if code:raise RuntimeError('Planning/collision control failed; see '+str(log))


if __name__=='__main__':main()
