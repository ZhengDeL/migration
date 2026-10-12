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
NPROC = {'worldengine': 4}
DATA_FILES = {'worldengine': ['data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl', 'data/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl', 'data/raw/openscene-v1.1/sensor_blobs/trainval', 'data/raw/openscene-v1.1/sensor_blobs/test', 'data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl', 'data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl', 'projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain_50pct.yaml', 'projects/AlgEngine/configs/navsim_splits/navtest_split/navtest.yaml', 'data/alg_engine/test_8192_kmeans.npy', 'data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth']}

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
    args = parser.parse_args()
    if args.dataset_version == 'v1.0-mini':
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
    devices = args.devices.split(',')
    if len(devices) != NPROC[args.model] or len(set(devices)) != len(devices) or any((not d.isdigit() for d in devices)):
        parser.error(f'{args.model} requires {NPROC[args.model]} distinct numeric device IDs')
    output = args.output_dir.resolve()
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = args.devices
    env['OMP_NUM_THREADS'] = env.get('OMP_NUM_THREADS', '8')
    legacy_smoke = args.smoke
    data_dir = None
    prepared = None
    if legacy_smoke:
        data_dir = (args.data_dir or ROOT / 'data' / 'worldengine_smoke').resolve()
        prepared = (args.prepared_dir or data_dir / 'prepared').resolve()
        names = ['worldengine_navtest_smoke.pkl', 'worldengine_navtest_smoke_pdm.pkl', 'worldengine_navtest_smoke.yaml']
        required = [prepared / name for name in ('manifest.json', *names)]
        from world_mini_data import ASSETS
        required += [data_dir / Path(path).name for (_, _, path, _, _) in ASSETS]
        required.append(source / LOCK['worldengine']['vocab_path'])
        missing = [str(path) for path in required if not path.is_file()]
    else:
        missing = [str(source / item) for item in DATA_FILES.get(args.model, ()) if not (source / item).exists()]
    if args.dataset_version == 'v1.0-mini':
        mini_scene = source / 'data/nuscenes/v1.0-mini/scene.json'
        if not mini_scene.is_file():
            missing.append(str(mini_scene))
    if legacy_smoke:
        cwd = source / 'projects/AlgEngine'
        env['WORLDENGINE_ROOT'] = str(source)
        config = output / f'{args.model}_real_smoke.py'
        if args.execute and (not missing):
            create = [sys.executable, str(ROOT / 'tools/create_legacy_mini_config.py'), args.model, '--repo', str(ROOT), '--source', str(source), '--data-dir', str(data_dir), '--prepared-dir', str(prepared), '--output-dir', str(output)]
            subprocess.run(create, env=env, check=True)
        entry = 'scripts/train.py'
        command = ['torchrun', '--standalone', '--nproc_per_node=4', entry, str(config), '--launcher', 'pytorch', '--work-dir', str(output), '--no-validate']
    else:
        settings = {'worldengine': ('scripts/train.py', 'configs/worldengine/e2e_vadv2_50pct.py', '5e-5', 8)}
        (entry, config, lr, epochs) = settings[args.model]
        cwd = source / 'projects/AlgEngine'
        env['WORLDENGINE_ROOT'] = str(source)
        epoch_count = 1 if args.smoke else epochs
        command = ['torchrun', '--standalone', f'--nproc_per_node={NPROC[args.model]}', entry, config, '--launcher', 'pytorch', '--work-dir', str(output)]
        if args.smoke:
            command.append('--no-validate')
        command += ['--cfg-options', 'dist_params.backend=hccl', f'optimizer.lr={lr}', 'data.samples_per_gpu=1', f'runner.max_epochs={epoch_count}', f'total_epochs={epoch_count}']
    env['PYTHONPATH'] = os.pathsep.join((part for part in (str(cwd), str(ROOT / 'tools'), env.get('PYTHONPATH', '')) if part))
    report = {'model': args.model, 'revision': LOCK[args.model]['revision'], 'patch_sha256': hashlib.sha256(patch_bytes(patch)).hexdigest() if patch else None, 'compat_patch_sha256': hashlib.sha256(patch_bytes(compat)).hexdigest() if compat else None, 'cwd': str(cwd), 'devices': args.devices, 'command': command, 'dataset_version': None, 'missing_data': missing, 'execute': args.execute}
    if legacy_smoke:
        report.update(prepared_dir=str(prepared), iterations=10, heldout_evaluation=False, dataset_scope=json.loads((prepared / 'manifest.json').read_text())['dataset_scope'] if (prepared / 'manifest.json').is_file() else 'data not prepared')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if missing:
        raise RuntimeError('required data or weight files are missing')
    check = [sys.executable, str(ROOT / 'tools/check_env.py'), '--model', args.model, '--nproc', str(NPROC[args.model]), '--output', str(output / 'environment.json')]
    subprocess.run(check, env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/check_msda.py'), '--output', str(output / 'msda.json')], env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(f'run failed: {exc}')
