"""Check model, optimizer, all rank RNG and next-data cursor after independently resumed training."""
import json
import sys
import torch
from common import own_path

a=torch.load(own_path(sys.argv[1],exists=True),map_location='cpu',weights_only=True)
b=torch.load(own_path(sys.argv[2],exists=True),map_location='cpu',weights_only=True)
counts={'tensors':0,'bit_equal_tensors':0,'max_absolute_error':0.0}
assert a['checkpoint_schema_version']==b['checkpoint_schema_version']==2
mismatches=[]
def compare(x,y,path):
    if isinstance(x,torch.Tensor):
        assert isinstance(y,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape,path
        counts['tensors']+=1;counts['bit_equal_tensors']+=int(torch.equal(x,y))
        if x.is_floating_point():
            counts['max_absolute_error']=max(counts['max_absolute_error'],float((x-y).abs().max()) if x.numel() else 0.0)
            if not torch.allclose(x,y,rtol=1e-5,atol=1e-6):
                mismatches.append({'path':path,'max_absolute_error':float((x-y).abs().max())})
        elif not torch.equal(x,y):mismatches.append({'path':path,'integer_state_equal':False})
    elif isinstance(x,dict):
        assert isinstance(y,dict) and set(x)==set(y),path
        for key in x:compare(x[key],y[key],path+'.'+str(key))
    elif isinstance(x,(list,tuple)):
        assert isinstance(y,(list,tuple)) and len(x)==len(y),path
        for index,(left,right) in enumerate(zip(x,y)):compare(left,right,path+'.'+str(index))
    elif x!=y:mismatches.append({'path':path,'metadata_equal':False})
for key in ['model','dataset_digest','world_size','step','state_dict','optimizer','resume_cursor','rank_rng']:
    compare(a[key],b[key],key)
compare(a['settings'].get('npu_precision'),b['settings'].get('npu_precision'),'settings.npu_precision')
compare(a['settings'].get('msda_backend'),b['settings'].get('msda_backend'),'settings.msda_backend')
compare(a['settings'].get('activation_profile'),b['settings'].get('activation_profile'),'settings.activation_profile')
compare(a['settings'].get('temporal_residual_profile'),b['settings'].get('temporal_residual_profile'),'settings.temporal_residual_profile')
compare(a['settings'].get('camera_projection_profile'),b['settings'].get('camera_projection_profile'),'settings.camera_projection_profile')
print(json.dumps({'result':'fail_continuous_vs_independently_resumed_training' if mismatches else 'pass_continuous_vs_independently_resumed_training','optimizer_steps':a['step'],
    'compared':'all model state and optimizer tensors; Python/NumPy/CPU/NPU RNG per rank; exact next epoch and batch cursor',
    'rtol':1e-5,'atol':1e-6,**counts,'mismatches':mismatches},indent=2),flush=True)
if mismatches:raise SystemExit(1)
