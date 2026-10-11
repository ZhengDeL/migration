#!/usr/bin/env python3
"""Apollo NPU forward/backward with synthetic inputs, independent of real datasets."""

import copy
import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
from prepare import require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True, type=Path)
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
source = args.source.resolve(strict=True)
out = args.output_dir.resolve()
require_checkout('apollo_visionnet', source)
patch = patch_for('apollo_visionnet', None)
compat = compat_patch_for('apollo_visionnet')
if patch_state(source, patch) != 'applied' or patch_state(source, compat) != 'applied':
    raise RuntimeError('Apply the pinned migration and compatibility patches first')
verify_applied_patch(source, patch, (compat,))
out.mkdir(parents=True, exist_ok=True)
os.chdir(source)
sys.path.insert(0, str(source))
rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world = int(os.environ.get('WORLD_SIZE', '1'))
device = torch.device(f'npu:{local_rank}')
torch.npu.set_device(local_rank)
torch.manual_seed(20261011)
report = {'model': 'apollo_visionnet', 'test': 'synthetic_input_forward_backward',
          'dataset_training': False, 'rank': rank, 'world_size': world,
          'image_shape': [1, 3, 6, 3, 96, 160], 'pretrained': False}
try:
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    from mmcv.device.npu import NPUDistributedDataParallel
    importlib.import_module('projects.mmdet3d_plugin')
    config_path = source / 'projects/configs/bevformer/bev_tiny_det_occ_apollo.py'
    cfg = Config.fromfile(str(config_path))
    report['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg.model.pretrained = None
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device).train()
    report['parameters'] = sum(p.numel() for p in model.parameters())
    if world > 1:
        dist.init_process_group('hccl')
        training_model = NPUDistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                   find_unused_parameters=True)
    else:
        training_model = model
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    watched = model.pts_bbox_head.query_embedding.weight
    before = watched.detach().clone()
    image = torch.randn(1, 3, 6, 3, 96, 160, device=device)
    projection = np.array([[1, 0, 80, 1600], [0, 1, 48, 960],
                           [0, 0, 1, 20], [0, 0, 0, 1]], dtype=np.float32)
    metadata = {i: {'can_bus': np.zeros(18, dtype=np.float32),
                    'lidar2img': [projection.copy() for _ in range(6)],
                    'img_shape': [(96, 160, 3)] * 6, 'pad_shape': [(96, 160, 3)] * 6,
                    'box_type_3d': LiDARInstance3DBoxes, 'scene_token': 'synthetic_fixture',
                    'prev_bev_exists': i > 0, 'scale_factor': 1.0, 'flip': False}
                for i in range(3)}
    box = LiDARInstance3DBoxes(torch.tensor([[0, 0, 0, 4, 2, 1.5, 0]], device=device), box_dim=7)
    label = torch.tensor([0], dtype=torch.long, device=device)
    occ = torch.tensor([[10000, 0], [10001, 1], [400000, 10]], dtype=torch.long, device=device)
    raw_checks = []
    def check_raw(module, inputs, outputs):
        if isinstance(outputs, dict):
            for name in ['all_cls_scores', 'all_bbox_preds', 'occupancy_preds']:
                value = outputs.get(name)
                if torch.is_tensor(value):
                    finite = bool(torch.isfinite(value).all())
                    raw_checks.append({'name': name, 'shape': list(value.shape), 'finite': finite})
                    if not finite:
                        raise RuntimeError(f'Non-finite raw head output: {name}')
    hook = model.pts_bbox_head.register_forward_hook(check_raw)
    losses = training_model(return_loss=True, img=image, img_metas=[copy.deepcopy(metadata)],
                            gt_bboxes_3d=[[box] * 3], gt_labels_3d=[[label] * 3], occ_gts=[[occ] * 3])
    hook.remove()
    report['losses'] = {k: float(v.detach().mean().item()) if torch.is_tensor(v) else float(v)
                        for k, v in losses.items()}
    loss = sum(v.mean() for k, v in losses.items() if 'loss' in k and torch.is_tensor(v))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Non-finite loss')
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError('Missing or non-finite gradients')
    report['gradient_tensors'] = len(gradients)
    report['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 35., foreach=False).item())
    optim.step()
    torch.npu.synchronize()
    report['parameter_changed'] = bool((before != watched.detach()).any())
    report['raw_outputs'] = raw_checks
    if not report['parameter_changed']:
        raise RuntimeError('Watched detection query parameter did not change')
    if rank == 0:
        checkpoint = out / 'fixture_step1.pth'
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'meta': {'synthetic_fixture': True, 'iterations': 1, 'world_size': world}}, checkpoint)
        report['checkpoint_bytes'] = checkpoint.stat().st_size
        report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if restored['meta']['synthetic_fixture'] is not True or not all(
            bool(torch.isfinite(t).all()) for t in restored['state_dict'].values()
            if t.is_floating_point()
        ):
            raise RuntimeError('Saved fixture checkpoint contains non-finite tensors')
        report['checkpoint_reload'] = 'pass'
    report['result'] = 'pass_compute_fixture'
except Exception as e:
    report['result'] = 'fail'
    report['failure'] = f'{type(e).__name__}: {e}'
    report['traceback'] = traceback.format_exc()
finally:
    (out / f'rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
raise SystemExit(0 if report['result'] == 'pass_compute_fixture' else 2)
