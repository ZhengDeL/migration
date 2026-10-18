"""Compare every validation and test prediction on CPU/NPU using identical selected weights."""
import argparse,json
from pathlib import Path
from common import PROFILE,own_path,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data_path',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=own_path(a.data_path,exists=True);cp=own_path(a.checkpoint,exists=True);output=own_path(a.output)
    if output.exists():raise RuntimeError('Preserve the previous complete-prediction evidence; choose a new output.')
    import numpy as np
    import torch,torch_npu
    from torch.utils.data import TensorDataset,DataLoader
    import model_task as task
    from full_training import metric_report
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch_npu.npu.set_device(0)
    device=torch.device('npu:0');metadata=task.check_data(data,'full')
    saved=torch.load(cp,map_location='cpu',weights_only=True)
    from npu_precision import checkpoint_precision
    precision=checkpoint_precision(torch_npu,saved['settings'])
    if saved['model']!=PROFILE['model'] or saved['dataset_digest']!=sha256(data/'metadata.json'):raise RuntimeError('Model/data digest differs.')
    cpu=task.create_model(metadata).eval();cpu.load_state_dict(saved['state_dict'],strict=True)
    npu=task.create_model(metadata).eval().to(device);npu.load_state_dict(saved['state_dict'],strict=True)
    splits={};passed=True
    for split in ['val','test']:
        arrays=np.load(data/(split+'.npz'),allow_pickle=False)
        dataset=TensorDataset(torch.from_numpy(arrays['x'].copy()),torch.from_numpy(arrays['y'].copy()))
        maximum=0.0;failed=elements=0;windows=0
        with torch.no_grad():
            for batch in DataLoader(dataset,batch_size=32,shuffle=False,drop_last=False):
                expected=task.predict(cpu,batch,torch.device('cpu'));actual=task.predict(npu,batch,device).cpu()
                if not bool(torch.isfinite(actual).all()):raise RuntimeError('Nonfinite complete-prediction output.')
                maximum=max(maximum,float((expected-actual).abs().max()))
                failed+=int((~torch.isclose(actual,expected,rtol=.003,atol=.003)).sum())
                elements+=actual.numel();windows+=len(batch[0])
        splits[split]={'windows':windows,'output_elements_compared':elements,'max_absolute_error_normalized':maximum,
            'elements_outside_tolerance':failed,'cpu_metrics':metric_report(cpu,dataset,metadata,torch.device('cpu'),32),
            'npu_metrics':metric_report(npu,dataset,metadata,device,32)}
        passed=passed and failed==0 and windows==metadata['counts'][split]
    result={'model':PROFILE['model'],'result':'pass' if passed else 'fail','checkpoint_sha256':sha256(cp),
        'data_scope':metadata['scope'],'npu_precision':precision,'prediction_rtol':.003,'prediction_atol':.003,'splits':splits,
        'validation_used_for_selection':True,'test_used_for_selection':False,'formal_business_accuracy_claimed':False}
    write_json(output,result);print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('Complete held-out CPU/NPU prediction comparison failed.')


if __name__=='__main__':main()
