#!/usr/bin/env python3
"""Run the original 24-epoch schedule on all mini training frames, then evaluate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    from mmcv import Config
    import mmcv
    from prepare import require_checkout, patch_for, compat_patch_for, verify_applied_patch
    source = ROOT / 'sources/BEVFormer'
    require_checkout('bevformer', source)
    patch = patch_for('bevformer', ROOT / 'vendor')
    verify_applied_patch(source, patch, (compat_patch_for('bevformer'),))
    train = mmcv.load(str(ROOT / 'prepared/nuscenes/nuscenes_infos_temporal_train.pkl'))
    val = mmcv.load(str(ROOT / 'prepared/nuscenes/nuscenes_infos_temporal_val.pkl'))
    if (len(train['infos']), len(val['infos'])) != (323, 81):
        raise RuntimeError('Expected every original mini train and validation frame')
    if {x['token'] for x in train['infos']} & {x['token'] for x in val['infos']}:
        raise RuntimeError('Mini train/val tokens overlap')
    initialization = source / 'ckpts/r101_dcn_fcos3d_pretrain.pth'
    if not initialization.is_file():
        raise RuntimeError('Provide the original FCOS3D backbone initialization')
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'mini_training.py').exists():
        raise RuntimeError('Use a fresh output directory inside this model')
    cfg = Config.fromfile(str(source / 'projects/configs/bevformer_fp16/bevformer_base_fp16.py'),
                          import_custom_modules=False)
    for split, annotation in [('train', 'train'), ('val', 'val'), ('test', 'val')]:
        cfg.data[split].ann_file = str(ROOT / 'prepared/nuscenes' / ('nuscenes_infos_temporal_' + annotation + '.pkl'))
        cfg.data[split].data_root = str(ROOT / 'data/nuscenes')
    cfg.load_from = str(initialization)
    cfg.resume_from = None
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 4
    cfg.dist_params.backend = 'hccl'
    cfg.optimizer.lr = 5e-5
    cfg.optimizer.foreach = False
    # The author's video runner accepts eval_model and keeps temporal iteration semantics.
    original_runner_type = cfg.runner.type
    cfg.runner.max_epochs = 24
    cfg.total_epochs = 24
    original_imports = (cfg.get('custom_imports') or {}).get('imports', [])
    if isinstance(original_imports, str):
        original_imports = [original_imports]
    cfg.custom_imports = dict(imports=[*original_imports, 'quality_training'], allow_failed_imports=False)
    cfg.custom_hooks = [dict(type='BEVMiniFiniteHook', priority='LOWEST')]
    cfg.checkpoint_config = dict(interval=1, save_optimizer=True, max_keep_ckpts=3)
    cfg.log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
    cfg.migration_training_scope = 'official mini 323 train / 81 val; original 24-epoch schedule; not full trainval'
    config = output / 'mini_training.py'
    cfg.dump(str(config))
    (output / 'mini_training_plan.json').write_text(json.dumps(dict(
        model='bevformer', train_frames=323, heldout_frames=81, train_val_disjoint=True,
        epochs=24, expected_updates=3888, nproc=2, batch_per_rank=1,
        lr=5e-5, original_runner_type=original_runner_type,
        fp16=dict(cfg.fp16), original_lr_schedule=dict(cfg.lr_config),
        initialization='Original FCOS3D R101-DCN pretraining; not author BEV trained weights',
        initialization_sha256=hashlib.sha256(initialization.read_bytes()).hexdigest(),
        task_quality_accepted=False, full_trainval_training_completed=False), indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), str(ROOT / 'tools'),
                                                    os.environ.get('PYTHONPATH', '')]))
    command = ['torchrun', '--standalone', '--nproc_per_node=2', 'tools/fp16/train.py', str(config),
               '--launcher', 'pytorch', '--work-dir', str(output), '--seed', '42', '--no-validate']
    result = subprocess.run(command, cwd=source, env=env)
    if result.returncode:
        return result.returncode
    return subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_mini.py'),
        '--output_dir', str(output), '--checkpoint', str(output / 'epoch_24.pth')], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
