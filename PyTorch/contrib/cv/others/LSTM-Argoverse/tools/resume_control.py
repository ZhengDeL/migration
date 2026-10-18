"""Compare full-curriculum continuous and independently restored epoch-boundary state."""
import json
import sys
import torch
from common import own_path

a=torch.load(own_path(sys.argv[1],exists=True),map_location='cpu',weights_only=True)
b=torch.load(own_path(sys.argv[2],exists=True),map_location='cpu',weights_only=True)
counts={'tensors':0,'bit_equal_tensors':0,'max_absolute_error':0.0}
def compare(x,y,path):
    if isinstance(x,torch.Tensor):
        assert isinstance(y,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape,path
        counts['tensors']+=1;counts['bit_equal_tensors']+=int(torch.equal(x,y))
        if x.is_floating_point():
            counts['max_absolute_error']=max(counts['max_absolute_error'],float((x-y).abs().max()) if x.numel() else 0.0)
            torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6,msg=path)
        else:assert torch.equal(x,y),path
    elif isinstance(x,dict):
        assert isinstance(y,dict) and set(x)==set(y),path
        for k in x:compare(x[k],y[k],path+'.'+str(k))
    elif isinstance(x,(tuple,list)):
        assert isinstance(y,(tuple,list)) and len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'.'+str(i))
    else:assert x==y,(path,x,y)
for key in ['model','world_size','dataset_digest','epoch','step','stage_index','state_dict','optimizer',
            'best_states','best_loss','previous_loss','increases','stage_epochs','rank_rng']:
    compare(a[key],b[key],key)
print(json.dumps({'result':'pass_uninterrupted_vs_epoch_boundary_resume','epochs':a['epoch'],
    'optimizer_steps':a['step'],'compared':'original encoder/decoder, both Adam states, curriculum and every rank RNG',
    'rtol':1e-5,'atol':1e-6,**counts},indent=2))
