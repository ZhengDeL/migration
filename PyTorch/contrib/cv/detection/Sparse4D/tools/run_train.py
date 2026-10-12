#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Build and optionally run one pinned Ascend training command."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from prepare import LOCK, ROOT, compat_patch_for, patch_for, patch_state, require_checkout, patch_bytes, verify_applied_patch, verify_sparse4d_files
NPROC = {'sparse4d': 2}
DATA_FILES = {'sparse4d': ['data/nuscenes_anno_pkls/nuscenes_infos_train.pkl', 'data/nuscenes_anno_pkls/nuscenes_infos_val.pkl', 'data/nuscenes', 'nuscenes_kmeans900.npy', 'ckpt/resnet50-19c8e357-modern.pth']}

def sparse_config(source: Path, output: Path, smoke: bool, dataset_version: str='v1.0-trainval') -> Path:
    original = source / 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'
    content = original.read_text(encoding='utf-8')
    substitutions = {'dist_params = dict(backend="nccl")': 'dist_params = dict(backend="hccl")', 'total_batch_size = 48': 'total_batch_size = 2', 'num_gpus = 8': 'num_gpus = 2', 'lr=6e-4,': 'lr=2.5e-5,', 'pretrained="ckpt/resnet50-19c8e357.pth"': 'pretrained="ckpt/resnet50-19c8e357-modern.pth"'}
    for (old, new) in substitutions.items():
        if content.count(old) != 1:
            raise RuntimeError(f'Sparse4D config changed: expected one {old!r}')
        content = content.replace(old, new)
    if dataset_version == 'v1.0-mini':
        old = 'version="v1.0-trainval"'
        if content.count(old) != 1:
            raise RuntimeError('Sparse4D config changed: expected one dataset version')
        content = content.replace(old, 'version="v1.0-mini"')
    if smoke:
        content += '\n# Only ten iterations for environment and backward smoke.\n'
        content += "runner = dict(type='IterBasedRunner', max_iters=10)\n"
        content += 'checkpoint_config = dict(interval=100000)\n'
        content += 'evaluation = dict(interval=100000, pipeline=vis_pipeline)\n'
    output.mkdir(parents=True, exist_ok=True)
    suffix = 'mini_smoke' if dataset_version == 'v1.0-mini' else 'smoke' if smoke else 'full'
    path = output / f'sparse4d_2p_{suffix}.py'
    path.write_text(content, encoding='utf-8')
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=NPROC)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--sdk', type=Path, help='required for BEVFormer and Sparse4D')
    parser.add_argument('--devices', required=True, help='logical NPU indices inside this process/container; map host physical cards before container launch')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true', help='ten optimizer steps or one epoch; no validation')
    parser.add_argument('--dataset-version', choices=('v1.0-trainval', 'v1.0-mini'), default='v1.0-trainval', help='nuScenes split; mini is for smoke only')
    parser.add_argument('--execute', action='store_true', help='run only after environment and data checks')
    parser.add_argument('--model-path', type=Path, help='local Qwen3-VL-8B checkpoint directory')
    parser.add_argument('--dataset-id', help='LLaMA-Factory dataset name from dataset_info.json')
    parser.add_argument('--data-dir', type=Path, help='Apollo/WorldEngine official asset directory')
    parser.add_argument('--prepared-dir', type=Path, help='verified Apollo/WorldEngine functional subset')
    parser.add_argument('--nuscenes', type=Path, help='raw nuScenes mini directory for Apollo')
    parser.add_argument('--anchor-file', type=Path, help='complete-data anchor; functional mode uses bundled mini anchor')
    args = parser.parse_args()
    if args.dataset_version == 'v1.0-mini' and (not args.smoke):
        parser.error('v1.0-mini is supported only for BEVFormer or Sparse4D --smoke')
    source = args.source.resolve(strict=True)
    sdk = args.sdk.resolve(strict=True) if args.sdk else None
    require_checkout(args.model, source)
    patch = patch_for(args.model, sdk)
    compat = compat_patch_for(args.model)
    if patch is not None and patch_state(source, patch) != 'applied':
        raise RuntimeError('apply the reviewed patch with tools/prepare.py --apply first')
    if compat is not None and patch_state(source, compat) != 'applied':
        raise RuntimeError('apply the compatibility patch with tools/prepare.py --apply first')
    if patch is not None:
        verify_applied_patch(source, patch, (compat,) if compat else ())
    if sdk is not None:
        verify_sparse4d_files(source, sdk)
    devices = args.devices.split(',')
    if len(devices) != NPROC[args.model] or len(set(devices)) != len(devices) or any((not d.isdigit() for d in devices)):
        parser.error(f'{args.model} requires {NPROC[args.model]} distinct numeric device IDs')
    output = args.output_dir.resolve()
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.devices
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '8')
    legacy_smoke = False
    data_dir = None
    prepared = None
    missing = [str(source / item) for item in DATA_FILES.get(args.model, ()) if not (source / item).exists()]
    if args.dataset_version == 'v1.0-mini':
        mini_scene = source / 'data/nuscenes/v1.0-mini/scene.json'
        if not mini_scene.is_file():
            missing.append(str(mini_scene))
    config = sparse_config(source, output, args.smoke, args.dataset_version)
    if args.anchor_file:
        anchor_path = args.anchor_file.resolve(strict=True)
        config.write_text(config.read_text().replace('anchor="nuscenes_kmeans900.npy"', 'anchor=' + repr(str(anchor_path))))
    command = ['torchrun', '--standalone', '--nproc_per_node=2', 'tools/train.py', str(config), '--launcher', 'pytorch', '--work-dir', str(output)]
    if args.smoke:
        command.append('--no-validate')
    cwd = source
    env['PYTHONPATH'] = os.pathsep.join((part for part in (str(cwd), str(ROOT / 'tools'), env.get('PYTHONPATH', '')) if part))
    report = {'model': args.model, 'revision': LOCK[args.model]['revision'], 'patch_sha256': hashlib.sha256(patch_bytes(patch)).hexdigest() if patch else None, 'compat_patch_sha256': hashlib.sha256(patch_bytes(compat)).hexdigest() if compat else None, 'cwd': str(cwd), 'devices': args.devices, 'command': command, 'dataset_version': args.dataset_version, 'missing_data': missing, 'execute': args.execute}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if missing:
        raise RuntimeError('required data or weight files are missing')
    check = [sys.executable, str(ROOT / 'tools/check_env.py'), '--model', args.model, '--nproc', str(NPROC[args.model]), '--output', str(output / 'environment.json')]
    subprocess.run(check, env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(f'run failed: {exc}')
