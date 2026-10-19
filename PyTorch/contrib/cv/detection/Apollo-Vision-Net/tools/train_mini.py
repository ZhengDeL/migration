#!/usr/bin/env python3
"""Train all real mini train frames for 24 epochs, then evaluate the heldout OCC split.

This is a local training diagnostic. Full trainval accuracy is a separate run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    from mmcv import Config
    from prepare import require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch
    from validate_legacy_mini import validate
    from verify_apollo_weight import verify
    source = ROOT / 'sources/Apollo-Vision-Net'
    require_checkout('apollo_visionnet', source)
    patch, compat = patch_for('apollo_visionnet', None), compat_patch_for('apollo_visionnet')
    if any(patch_state(source, p) != 'applied' for p in (patch, compat)):
        raise RuntimeError('Apply the pinned Apollo patches first')
    verify_applied_patch(source, patch, (compat,))
    prepared = ROOT / 'prepared/apollo'
    manifest = validate('apollo_visionnet', prepared, ROOT / 'data/apollo')
    if (manifest['train_samples'], manifest['heldout_samples']) != (242, 162):
        raise RuntimeError('Expected the complete original OCC/mini intersection')
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'mini_training.py').exists():
        raise RuntimeError('Use a fresh output directory inside this model')
    initialization = verify(source)
    cfg = Config.fromfile(str(source / 'projects/configs/bevformer/bev_tiny_det_occ_apollo.py'),
                          import_custom_modules=False)
    cfg.model.pretrained = dict(img=str(initialization))
    cfg.load_from = None
    cfg.resume_from = None
    for split, annotation in [('train', 'apollo_mini_train.pkl'), ('val', 'apollo_mini_val.pkl'),
                              ('test', 'apollo_mini_val.pkl')]:
        cfg.data[split].ann_file = str(prepared / annotation)
        cfg.data[split].data_root = str(ROOT / 'data/nuscenes')
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 4
    cfg.dist_params.backend = 'hccl'
    cfg.optimizer.lr = 1e-4  # Original 2e-4 at eight cards, linearly scaled to four.
    cfg.optimizer.foreach = False
    cfg.custom_imports = dict(imports=['quality_training'], allow_failed_imports=False)
    cfg.optimizer_config = dict(type='ApolloMiniQualityOptimizerHook',
        model_name='apollo_visionnet', dataset_scope=manifest['dataset_scope'],
        initialization_checkpoint=str(initialization), grad_clip=dict(max_norm=35, norm_type=2))
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=24)
    cfg.total_epochs = 24
    cfg.checkpoint_config = dict(interval=1, save_optimizer=True, max_keep_ckpts=3)
    cfg.log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
    cfg.migration_training_scope = 'original OCC mini split: 242 train / 162 heldout; not full trainval'
    path = output / 'mini_training.py'
    cfg.dump(str(path))
    plan = dict(model='apollo_visionnet', scope=cfg.migration_training_scope,
        train_frames=242, heldout_frames=162, epochs=24, nproc=4, batch_per_rank=1,
        expected_updates=24 * ((242 + 3) // 4), initialization='official DLA depth backbone only',
        formal_trainval_training_completed=False, task_quality_accepted=False,
        original_config='projects/configs/bevformer/bev_tiny_det_occ_apollo.py',
        retained_learning_rate_schedule=dict(cfg.lr_config))
    (output / 'mini_training_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), str(ROOT / 'tools'),
                                                   os.environ.get('PYTHONPATH', '')]))
    command = ['torchrun', '--standalone', '--nproc_per_node=4', 'tools/train.py', str(path),
               '--launcher', 'pytorch', '--work-dir', str(output), '--seed', '42', '--no-validate']
    result = subprocess.run(command, cwd=source, env=env)
    if result.returncode:
        return result.returncode
    # The original OCC mini split has 162 frames; use its original OCC evaluator.
    return subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_mini.py'),
        '--output_dir', str(output), '--checkpoint', str(output / 'epoch_24.pth')], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
