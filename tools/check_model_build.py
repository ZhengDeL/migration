#!/usr/bin/env python3
"""Check pinned Apollo or AlgEngine imports, registry construction and NPU placement."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

from prepare import LOCK, require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('worldengine', 'apollo_visionnet'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    initial_cwd = Path.cwd()
    report = {'model': args.model, 'revision': LOCK[args.model]['revision'],
              'dataset_training': False, 'pretrained_loaded': False}
    try:
        require_checkout(args.model, source)
        patch = patch_for(args.model, None)
        compat = compat_patch_for(args.model)
        if patch_state(source, patch) != 'applied' or (compat and patch_state(source, compat) != 'applied'):
            raise RuntimeError('Apply the pinned migration and compatibility patches first')
        verify_applied_patch(source, patch, (compat,) if compat else ())
        if args.model == 'worldengine':
            work = source / 'projects/AlgEngine'
            relative_config = 'configs/worldengine/e2e_vadv2_50pct.py'
            plugin = 'mmdet3d_plugin'
            os.environ['WORLDENGINE_ROOT'] = str(source)
        else:
            work = source
            relative_config = 'projects/configs/bevformer/bev_tiny_det_occ_apollo.py'
            plugin = 'projects.mmdet3d_plugin'
        os.chdir(work)
        sys.path.insert(0, str(work))
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        from mmcv import Config
        from mmdet3d.models import build_model
        torch.npu.set_device(0)
        cfg = Config.fromfile(str(work / relative_config))
        importlib.import_module(plugin)
        report['import'] = 'pass'
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        report['build'] = 'pass'
        model = model.to('npu:0')
        if not all(p.device.type == 'npu' for p in model.parameters()):
            raise RuntimeError('Some model parameters were not placed on NPU')
        report['npu_placement'] = 'pass'
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['result'] = 'pass_model_build'
    except Exception as exc:
        report['result'] = 'fail'
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    finally:
        os.chdir(initial_cwd)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['result'] == 'pass_model_build' else 2


if __name__ == '__main__':
    raise SystemExit(main())
