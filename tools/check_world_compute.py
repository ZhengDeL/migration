#!/usr/bin/env python3
"""AlgEngine VADv2 NPU forward/backward with clearly marked synthetic inputs."""

import argparse
import copy
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
from fetch_world_vocab import verified
from prepare import LOCK, require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    report = {'model': 'worldengine', 'scope': 'AlgEngine VADv2 50pct',
              'test': 'synthetic_input_forward_backward', 'dataset_training': False,
              'rank': rank, 'world_size': world, 'image_shape': [1, 4, 8, 3, 96, 160],
              'bev_shape': [200, 200], 'pretrained': False}
    try:
        require_checkout('worldengine', source)
        patch = patch_for('worldengine', None)
        compat = compat_patch_for('worldengine')
        if patch_state(source, patch) != 'applied' or patch_state(source, compat) != 'applied':
            raise RuntimeError('Apply the pinned migration and compatibility patches first')
        verify_applied_patch(source, patch, (compat,))
        vocab_path = source / LOCK['worldengine']['vocab_path']
        if not verified(vocab_path):
            raise RuntimeError('Download and SHA256-verify the pinned official trajectory vocabulary first')
        report['vocab_sha256'] = LOCK['worldengine']['vocab_sha256']
        work = source / 'projects/AlgEngine'
        os.environ['WORLDENGINE_ROOT'] = str(source)
        os.chdir(work)
        sys.path.insert(0, str(work))
        from mmcv import Config
        from mmcv.device.npu import NPUDistributedDataParallel
        from mmdet3d.models import build_model
        importlib.import_module('mmdet3d_plugin')
        config_path = work / 'configs/worldengine/e2e_vadv2_50pct.py'
        cfg = Config.fromfile(str(config_path))
        report['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if cfg.model.get('process_perception', False) or cfg.model.planning_head.reward_shaping:
            raise RuntimeError('This fixture covers the pinned imitation-learning planning configuration')
        torch.npu.set_device(local_rank)
        device = torch.device(f'npu:{local_rank}')
        torch.manual_seed(20261011)
        cfg.model.pretrained = None
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        model = model.to(device).train()
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['trajectory_count'] = model.planning_head.vocab.shape[0]
        if world > 1:
            dist.init_process_group('hccl')
            training_model = NPUDistributedDataParallel(
                model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        else:
            training_model = model
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=5e-5, foreach=False)
        watched = model.planning_head.heads['imi'][-1].weight
        before = watched.detach().clone()
        projection = np.array([[1, 0, 80, 1600], [0, 1, 48, 960],
                               [0, 0, 1, 20], [0, 0, 0, 1]], dtype=np.float32)
        metadata = {i: {'can_bus': np.zeros(18, dtype=np.float32),
                        'lidar2img': [projection.copy() for _ in range(8)],
                        'lidar2global_rotation': np.eye(3, dtype=np.float32),
                        'img_shape': [(96, 160, 3)] * 8, 'pad_shape': [(96, 160, 3)] * 8,
                        'scene_token': 'synthetic_fixture', 'prev_bev_exists': i > 0}
                    for i in range(4)}
        raw_checks = []
        original_forward = model.planning_head.forward
        original_loss = model.planning_head.loss

        def check_tensor(name, value):
            finite = bool(torch.isfinite(value).all())
            raw_checks.append({'name': name, 'shape': list(value.shape), 'finite': finite})
            if not finite:
                raise RuntimeError(f'Non-finite raw planning tensor: {name}')

        def checked_forward(bev_embed, *inputs, **kwargs):
            check_tensor('bev_embed', bev_embed)
            result = original_forward(bev_embed, *inputs, **kwargs)
            for name, value in result.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    check_tensor(name, value)
            return result

        def checked_loss(*inputs, **kwargs):
            losses = original_loss(*inputs, **kwargs)
            # NAVFormer later applies nan_to_num: check the original loss before that operation.
            for name, value in losses.items():
                check_tensor('raw_' + name, value)
            return losses

        model.planning_head.forward = checked_forward
        model.planning_head.loss = checked_loss
        torch.manual_seed(20261011 + rank)
        image = torch.randn(1, 4, 8, 3, 96, 160, device=device)
        # Use one genuine vocabulary trajectory as a synthetic imitation target.
        future = model.planning_head.vocab[0, 4::5].detach().view(1, 1, 8, 3)
        losses = training_model(
            return_loss=True, img=image, img_metas=[copy.deepcopy(metadata)],
            l2g_r_mat=torch.eye(3, device=device).view(1, 1, 3, 3).repeat(1, 4, 1, 1),
            l2g_t=torch.zeros(1, 4, 3, device=device),
            timestamp=torch.arange(4, device=device).float().view(1, 4),
            sdc_planning=future, sdc_planning_mask=torch.ones(1, 1, 8, 3, device=device),
            command=torch.zeros(1, dtype=torch.long, device=device),
            sdc_planning_past=torch.zeros(1, 1, 4, 4, device=device),
            sdc_planning_mask_past=torch.ones(1, 1, 4, 4, device=device),
            gt_pre_command_sdc=torch.zeros(1, 1, 4, 1, dtype=torch.long, device=device),
            sdc_status=torch.zeros(1, 4, device=device))
        report['losses'] = {k: float(v.detach().mean().item()) for k, v in losses.items()}
        loss = sum(v.mean() for k, v in losses.items() if 'loss' in k)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Missing or non-finite gradients')
        report['gradient_tensors'] = len(gradients)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35., foreach=False)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError('Non-finite gradient norm')
        report['gradient_norm'] = float(gradient_norm.item())
        optimizer.step()
        torch.npu.synchronize()
        report['parameter_changed'] = bool((before != watched.detach()).any())
        report['raw_outputs'] = raw_checks
        if not report['parameter_changed']:
            raise RuntimeError('Watched imitation-scoring parameter did not change')
        if world > 1:
            gathered = [torch.empty_like(watched) for _ in range(world)]
            dist.all_gather(gathered, watched.detach().contiguous())
            if not all(torch.allclose(watched.detach(), item, atol=1e-6, rtol=1e-5) for item in gathered):
                raise RuntimeError('Watched parameters differ between ranks after DDP update')
            report['parameter_sync'] = 'pass'
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
    except Exception as exc:
        report['result'] = 'fail'
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    finally:
        (out / f'rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0 if report['result'] == 'pass_compute_fixture' else 2


if __name__ == '__main__':
    raise SystemExit(main())
