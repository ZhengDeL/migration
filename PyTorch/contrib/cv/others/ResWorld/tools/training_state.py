"""Model-local complete training cursor and per-rank RNG checkpoint support."""
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def capture_rng(torch_npu,local_rank):
    state=np.random.get_state()
    return {'torch':torch.get_rng_state(),'npu':torch_npu.npu.get_rng_state(local_rank).cpu(),
            'python':random.getstate(),'numpy':{'algorithm':state[0],
                'keys':torch.tensor(state[1].astype(np.int64)),'position':int(state[2]),
                'has_gauss':int(state[3]),'cached_gaussian':float(state[4])}}

def restore_rng(value,torch_npu,local_rank):
    torch.set_rng_state(value['torch']);torch_npu.npu.set_rng_state(value['npu'],device=local_rank)
    random.setstate(value['python'])
    state=value['numpy']
    np.random.set_state((state['algorithm'],np.asarray(state['keys'].tolist(),dtype=np.uint32),
        state['position'],state['has_gauss'],state['cached_gaussian']))

def epoch_loader(original,epoch,rank,skip,remaining):
    batches=list(original.batch_sampler)
    if skip<0 or skip>len(batches):raise RuntimeError('Resume batch cursor is outside this epoch.')
    selected=batches[skip:skip+remaining if remaining else None]
    # Iterating the DataLoader must not consume the CPU RNG used by augmentations.
    generator=torch.Generator().manual_seed(910000+epoch*64+rank)
    loader=DataLoader(original.dataset,batch_sampler=selected,num_workers=0,
        collate_fn=original.collate_fn,generator=generator)
    return loader,len(batches)

def batch_digest(value):
    digest=hashlib.sha256()
    def add(item):
        if isinstance(item,torch.Tensor):
            x=item.detach().cpu().contiguous()
            digest.update(str((x.dtype,tuple(x.shape))).encode());digest.update(x.numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item):digest.update(str(key).encode());add(item[key])
        elif isinstance(item,(tuple,list)):
            for entry in item:add(entry)
        else:digest.update(json.dumps(item,sort_keys=True,default=str).encode())
    add(value);return digest.hexdigest()
