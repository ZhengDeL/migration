#!/usr/bin/env python3
"""Run the pinned Sparse4D 100 epoch schedule on the available mini training split.

Two cards keep the original per-card batch 6; lr scales from batch 48 to 12.
This is diagnostic mini training, followed by all 81 heldout detection frames.
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
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    from mmcv import Config
    from prepare import require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch, verify_sparse4d_files
    from run_train import sparse_config
    source, sdk = ROOT / 'sources/Sparse4D', ROOT / 'vendor'
    require_checkout('sparse4d', source)
    patch, compat = patch_for('sparse4d', sdk), compat_patch_for('sparse4d')
    if any(patch_state(source, p) != 'applied' for p in (patch, compat) if p):
        raise RuntimeError('Apply the pinned patches first')
    verify_applied_patch(source, patch, (compat,) if compat else ())
    verify_sparse4d_files(source, sdk)
    import mmcv
    train = mmcv.load(str(ROOT / 'prepared/nuscenes/nuscenes_infos_train.pkl'))
    val = mmcv.load(str(ROOT / 'prepared/nuscenes/nuscenes_infos_val.pkl'))
    if (len(train['infos']), len(val['infos'])) != (323, 81):
        raise RuntimeError('Expected the complete official mini split')
    if {x['token'] for x in train['infos']} & {x['token'] for x in val['infos']}:
        raise RuntimeError('Mini train and validation overlap')
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'mini_training.py').exists():
        raise RuntimeError('Use a fresh output directory inside this model')
    path = sparse_config(source, output, False, 'v1.0-mini')
    text = path.read_text()
    for old, new in [('total_batch_size = 2', 'total_batch_size = 12'),
                     ('lr=2.5e-5,', 'lr=1.5e-4,'),
                     ('num_iters_per_epoch = int(28130 // (num_gpus * batch_size))',
                      'num_iters_per_epoch = int(323 // (num_gpus * batch_size))')]:
        if text.count(old) != 1:
            raise RuntimeError('Pinned mini configuration changed: ' + old)
        text = text.replace(old, new)
    path.write_text(text)
    cfg = Config.fromfile(str(path))
    if cfg.runner.max_iters != 2600 or cfg.data.samples_per_gpu != 6:
        raise RuntimeError('Unexpected original schedule adapted to mini')
    cfg.custom_imports = dict(imports=['quality_training'], allow_failed_imports=False)
    cfg.custom_hooks = [dict(type='SparseMiniFiniteHook', priority='LOWEST')]
    cfg.checkpoint_config = dict(interval=260, save_optimizer=True, max_keep_ckpts=3)
    cfg.log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
    cfg.migration_training_scope = 'official mini 323 train / 81 val; 100 iteration-defined mini epochs; not full trainval'
    config = output / 'mini_training.py'
    cfg.dump(str(config))
    (output / 'mini_training_plan.json').write_text(json.dumps(dict(
        model='sparse4d', train_frames=323, heldout_frames=81, train_val_disjoint=True,
        iteration_defined_epochs=100, updates=2600, nproc=2, batch_per_rank=6,
        lr=1.5e-4, initialization='original ImageNet ResNet50 backbone only',
        formal_trainval_training_completed=False, task_quality_accepted=False), indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), str(ROOT / 'tools'),
                                                    os.environ.get('PYTHONPATH', '')]))
    command = ['torchrun', '--standalone', '--nproc_per_node=2', 'tools/train.py', str(config),
               '--launcher', 'pytorch', '--work-dir', str(output), '--seed', '42', '--no-validate']
    result = subprocess.run(command, cwd=source, env=env)
    if result.returncode:
        return result.returncode
    return subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_mini.py'),
        '--output_dir', str(output), '--checkpoint', str(output / 'iter_2600.pth')], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
