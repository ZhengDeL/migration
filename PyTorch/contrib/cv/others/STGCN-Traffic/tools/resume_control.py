"""Compare uninterrupted and epoch-boundary resumed training without exposing test data."""
import json
import sys
import torch
from common import own_path

left = torch.load(own_path(sys.argv[1], exists=True), map_location='cpu', weights_only=True)
right = torch.load(own_path(sys.argv[2], exists=True), map_location='cpu', weights_only=True)
counts = {'tensors': 0, 'bit_equal_tensors': 0, 'max_absolute_error': 0.0}

def compare(a, b, path):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, path
        counts['tensors'] += 1
        equal = torch.equal(a, b)
        counts['bit_equal_tensors'] += int(equal)
        if a.is_floating_point():
            error = float((a-b).abs().max()) if a.numel() else 0.0
            counts['max_absolute_error'] = max(counts['max_absolute_error'], error)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=path)
        else:
            assert equal, path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:compare(a[key], b[key], path+'.'+str(key))
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for index, (x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(index))
    else:assert a == b, (path, a, b)

for name in ['model','world_size','dataset_digest','epochs_completed','step','state_dict',
             'optimizer','scheduler','best_state_dict','best_epoch','bad_epochs','rank_rng']:
    compare(left[name], right[name], name)
print(json.dumps({'result':'pass_uninterrupted_vs_epoch_boundary_resume',
    'epochs':left['epochs_completed'],'optimizer_steps':left['step'],
    'compared':'all model tensors, optimizer, scheduler, selected-best model, counters and every rank RNG',
    'rtol':1e-5,'atol':1e-6,**counts},indent=2))
