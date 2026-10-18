"""Evaluate the pinned author's planning and collision metrics on every held-out mini frame.

Run inside this model's pinned container with one visible NPU. Mini results do not
substitute for nuScenes trainval acceptance or reproduction of the published paper.
"""
import argparse,copy,json
from pathlib import Path
from common import ROOT,own_path,write_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path',type=Path,default=ROOT/'data/nuscenes')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--hf32',choices=['runtime_default','off'],default='off')
    args=parser.parse_args();data=own_path(args.data_path,exists=True);checkpoint=own_path(args.checkpoint,exists=True)
    output=own_path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch,torch_npu
    from mmcv.parallel import DataContainer
    import model_task as task
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if args.hf32=='off':
        torch_npu.npu.conv.allow_hf32=False;torch_npu.npu.matmul.allow_hf32=False
        if hasattr(torch_npu.npu,'aclnn'):torch_npu.npu.aclnn.allow_hf32=False
    torch_npu.npu.set_device(0);device=torch.device('npu:0')
    metadata=task.check_data(data,'functional')
    sources=json.loads((ROOT/'pretrained_sources.lock.json').read_text())
    locked=sources['files'].get(checkpoint.name)
    if locked and (sha256(checkpoint)!=locked['sha256'] or checkpoint.stat().st_size!=locked['bytes']):
        raise RuntimeError('Locked pretrained asset digest/length differs.')
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'dataset_digest' in raw and raw['dataset_digest']!=sha256(data/'metadata.json'):
        raise RuntimeError('The project checkpoint was trained with a different data manifest.')
    state=raw.get('state_dict',raw)
    if state and all(key.startswith('module.') for key in state):state={key[7:]:value for key,value in state.items()}
    cpu=task.create_model(metadata).cpu().eval();cpu.load_state_dict(state,strict=True)
    npu=task.create_model(metadata).to(device).eval();npu.load_state_dict(state,strict=True)
    _,dataset=task.datasets(data,metadata)
    def one(value):
        while True:
            if isinstance(value,DataContainer):value=value.data
            elif isinstance(value,(list,tuple)) and len(value)==1:value=value[0]
            else:return value
    rows=[];totals={'cpu':{},'npu':{}};passed=True
    with torch.no_grad():
        for index in range(len(dataset)):
            example=dataset[index];sample=example['sample'];batch=task.collate([example])
            pred_cpu=task.predict(cpu,batch,torch.device('cpu')).cpu()
            pred_npu=task.predict(npu,batch,device).cpu()
            close=bool(torch.isclose(pred_cpu,pred_npu,rtol=.003,atol=.003).all())
            if not bool(torch.isfinite(pred_npu).all()):raise RuntimeError('Nonfinite complete planning output.')
            command=batch['ego_fut_cmd'].reshape(-1,3)
            if not bool((command.sum(-1)==1).all()):raise RuntimeError('Official navigation annotation must select one command.')
            chosen=int(command.argmax(-1)[0]);truth=batch['ego_fut_trajs'].reshape(1,6,2).cumsum(1)
            if not bool((batch['ego_fut_masks'].reshape(1,6)==1).all()):raise RuntimeError('Only complete future labels are allowed.')
            annotations={'gt_agent_boxes':one(sample['gt_bboxes_3d']),
                'gt_agent_feats':one(sample['gt_attr_labels']).unsqueeze(0).cpu(),
                'gt_map_boxes':one(sample['map_gt_bboxes_3d']),
                'gt_map_labels':one(sample['map_gt_labels_3d']).cpu()}
            original_boxes=annotations['gt_agent_boxes'].tensor.clone()
            metric={}
            for name,prediction in [('cpu',pred_cpu),('npu',pred_npu)]:
                # The original rasterizer mutates a numpy view of box yaw. Each
                # backend must receive independent annotations, including boxes.
                actual=cpu.compute_planner_metric_stp3(pred_ego_fut_trajs=prediction[:,chosen].cumsum(1),
                    gt_ego_fut_trajs=truth,**copy.deepcopy(annotations),fut_valid_flag=True)
                metric[name]={key:float(value) for key,value in actual.items() if key!='fut_valid_flag'}
                for key,value in metric[name].items():totals[name][key]=totals[name].get(key,0.0)+value
            if not torch.equal(original_boxes,annotations['gt_agent_boxes'].tensor):raise RuntimeError('Shared original annotations were mutated.')
            equal={key:bool(np.isclose(value,metric['npu'][key],rtol=.003,atol=.003)) if 'L2' in key
                   else value==metric['npu'][key] for key,value in metric['cpu'].items()}
            passed=passed and close and all(equal.values())
            rows.append({'token':example['token'],'prediction_close':close,
                'prediction_max_absolute_error':float((pred_cpu-pred_npu).abs().max()),
                'metric_comparison':equal,'metrics':metric})
            print(json.dumps({'frames_evaluated':index+1,'expected':len(dataset),'token':example['token']}),flush=True)
    if len(rows)!=65 or len(set(row['token'] for row in rows))!=65:raise RuntimeError('Expected all 65 distinct held-out mini frames.')
    from npu_activations import activation_profile, activation_execution_counts
    from temporal_residual import TEMPORAL_RESIDUAL_PROFILE
    result={'model':'resworld','result':'pass_original_mini_planning_collision_control' if passed else 'fail',
        'checkpoint_sha256':sha256(checkpoint),'data_scope':metadata['scope'],'held_out_frames':len(rows),
        'checkpoint_source':locked['url'] if locked else 'project or user supplied model-local checkpoint',
        'pretraining_overlap_with_mini_not_excluded':bool(locked),
        'precision_variant':args.hf32,'conv_allow_hf32':torch_npu.npu.conv.allow_hf32,
        'matmul_allow_hf32':torch_npu.npu.matmul.allow_hf32,
        'official_metric_function':'ResWorld.compute_planner_metric_stp3 from pinned vendor',
        'metric_source_sha256':sha256(ROOT/'vendor/ResWorld/projects/mmdet3d_plugin/resworld/planner/metric_stp3.py'),
        'metrics':{backend:{key:value/len(rows) for key,value in values.items()} for backend,values in totals.items()},
        'predictions_rtol':.003,'predictions_atol':.003,'collision_gate':'identical per-frame original metric values',
        'independent_ground_truth_copies_verified':True,'rows':rows,
        'navigation_command':'official annotation; not predicted','formal_full_trainval_accuracy_claimed':False}
    result['activation_profile']=activation_profile()
    result['activation_execution_counts']=activation_execution_counts()
    result['temporal_residual_profile']=TEMPORAL_RESIDUAL_PROFILE
    from camera_geometry import CAMERA_PROJECTION_PROFILE, projection_execution_counts
    result['camera_projection_profile']=CAMERA_PROJECTION_PROFILE
    result['projection_execution_counts']=projection_execution_counts()
    if projection_execution_counts()['npu_pipeline']==0:
        raise RuntimeError('The constant camera projection did not execute in the NPU pipeline.')
    write_json(output/'planning_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not passed:raise RuntimeError('Original planning/collision CPU/NPU control failed; all actual metrics preserved.')


if __name__=='__main__':main()
