#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Generate a ten-update real-data config from each pinned upstream config."""
import argparse
import json
import sys
from pathlib import Path
from mmcv import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['apollo_visionnet'])
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--nuscenes', type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    sys.path.insert(0, str(args.repo / 'tools'))
    from prepare import require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch
    require_checkout(args.model, source)
    patch = patch_for(args.model, None)
    compat = compat_patch_for(args.model)
    if patch_state(source, patch) != 'applied' or patch_state(source, compat) != 'applied':
        raise RuntimeError('Pinned source patches must be applied first')
    verify_applied_patch(source, patch, (compat,))
    prepared = args.prepared_dir.resolve(strict=True) if args.prepared_dir else args.data_dir.resolve(strict=True) / 'prepared'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    from validate_legacy_mini import validate
    validate(args.model, prepared, args.data_dir)
    if manifest['model'] != args.model:
        raise RuntimeError('Data manifest model mismatch')
    cfg = Config.fromfile(str(source / 'projects/configs/bevformer/bev_tiny_det_occ_apollo.py'), import_custom_modules=False)
    from verify_apollo_weight import verify as verify_apollo_weight
    initialization = verify_apollo_weight(source)
    cfg.model.pretrained = dict(img=str(initialization))
    cfg.load_from = None
    cfg.data.train.ann_file = str(prepared / 'apollo_mini_train.pkl')
    cfg.data.train.data_root = str(args.nuscenes.resolve(strict=True))
    cfg.data.val.ann_file = str(prepared / 'apollo_mini_val.pkl')
    cfg.data.val.data_root = cfg.data.train.data_root
    cfg.data.test = cfg.data.val.copy()
    cfg.custom_imports = dict(imports=['finite_training'], allow_failed_imports=False)
    scope = manifest['dataset_scope'] + '; official DLA depth pretrained backbone'
    lr = 0.0001
    cfg.pop('total_epochs', None)
    cfg.pop('fp16', None)
    cfg.dist_params.backend = 'hccl'
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.optimizer.lr = lr
    cfg.optimizer.foreach = False
    cfg.optimizer_config = dict(type='MigrationFiniteOptimizerHook', model_name=args.model, dataset_scope=scope, expected_iterations=10, grad_clip=dict(max_norm=35, norm_type=2))
    cfg.optimizer_config.initialization_checkpoint = str(initialization)
    cfg.lr_config = dict(policy='CosineAnnealing', by_epoch=False, warmup='linear', warmup_iters=2, warmup_ratio=1.0 / 3, min_lr_ratio=0.001)
    cfg.runner = dict(type='IterBasedRunner', max_iters=10)
    cfg.checkpoint_config = dict(by_epoch=False, interval=10, save_optimizer=True, max_keep_ckpts=1)
    cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    cfg.workflow = [('train', 1)]
    cfg.evaluation.interval = 100000
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output / f'{args.model}_real_smoke.py'
    cfg.dump(str(config))
    print(json.dumps({'config': str(config), 'iterations': 10, 'dataset_scope': scope, 'load_from': cfg.load_from, 'initialization_checkpoint': cfg.optimizer_config.initialization_checkpoint, 'image_pipeline': cfg.data.train.pipeline[0]}), flush=True)
if __name__ == '__main__':
    main()
