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
NPROC = {'qwen3vl8b': 2}
DATA_FILES = {}

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
    parser.add_argument('--data-dir', type=Path, help='Verified model-local data/drivelm directory for full training')
    parser.add_argument('--prepared-dir', type=Path, help='verified Apollo/WorldEngine functional subset')
    parser.add_argument('--nuscenes', type=Path, help='raw nuScenes mini directory for Apollo')
    args = parser.parse_args()
    protocol = None
    if not args.smoke:
        from prepare_full_data import require_full_protocol
        protocol = require_full_protocol(args.data_dir, args.dataset_id)
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
    legacy_smoke = False
    data_dir = None
    prepared = None
    missing = [str(source / item) for item in DATA_FILES.get(args.model, ()) if not (source / item).exists()]
    if args.dataset_version == 'v1.0-mini':
        mini_scene = source / 'data/nuscenes/v1.0-mini/scene.json'
        if not mini_scene.is_file():
            missing.append(str(mini_scene))
    if args.model_path is None or args.dataset_id is None:
        parser.error('Qwen3-VL requires --model-path and --dataset-id')
    model_dir = args.model_path.resolve()
    dataset_base = source / 'data' if args.smoke else Path(protocol['dataset_dir'])
    dataset_registry = dataset_base / 'dataset_info.json'
    if not model_dir.is_dir():
        missing.append(str(model_dir))
    else:
        if not (model_dir / 'config.json').is_file():
            missing.append(str(model_dir / 'config.json'))
        hf_marker = model_dir / '.source_revision'
        ms_marker = model_dir / '.source_modelscope_revision'
        hf_verified = hf_marker.is_file() and hf_marker.read_text(encoding='utf-8').strip() == LOCK['qwen3vl8b']['checkpoint_revision']
        ms_verified = ms_marker.is_file() and ms_marker.read_text(encoding='utf-8').strip() == LOCK['qwen3vl8b']['checkpoint_modelscope_revision']
        if not (hf_verified or ms_verified):
            missing.append(f'{hf_marker} or {ms_marker}: pinned download verification required')
        for asset in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'preprocessor_config.json', 'video_preprocessor_config.json'):
            if not (model_dir / asset).is_file():
                missing.append(str(model_dir / asset))
        weight_index = model_dir / 'model.safetensors.index.json'
        if weight_index.is_file():
            index = json.loads(weight_index.read_text(encoding='utf-8'))
            weight_map = index.get('weight_map')
            if not isinstance(weight_map, dict) or not weight_map:
                raise RuntimeError('Qwen model weight index has no weight map')
            for shard in set(weight_map.values()):
                if not (model_dir / shard).is_file() or (model_dir / shard).stat().st_size < 1024 * 1024:
                    missing.append(str(model_dir / shard))
        else:
            missing.append(str(weight_index))
    if not dataset_registry.is_file():
        missing.append(str(dataset_registry))
    else:
        registered = json.loads(dataset_registry.read_text(encoding='utf-8'))
        if args.dataset_id not in registered:
            missing.append(f'dataset-id:{args.dataset_id}')
        else:
            dataset_info = registered[args.dataset_id]
            file_name = dataset_info.get('file_name') if isinstance(dataset_info, dict) else None
            if file_name and (not (dataset_base / file_name).is_file()):
                missing.append(str(dataset_base / file_name))
    if args.execute and (not args.model_path.is_dir()):
        raise RuntimeError(f'checkpoint directory missing: {args.model_path}')
    if args.execute and (not (args.model_path / 'config.json').is_file()):
        raise RuntimeError('Qwen checkpoint config.json missing')
    if args.execute and (not dataset_registry.is_file()):
        raise RuntimeError('LLaMA-Factory data/dataset_info.json missing')
    if args.execute:
        model_config = json.loads((args.model_path / 'config.json').read_text(encoding='utf-8'))
        if model_config.get('model_type') != 'qwen3_vl':
            raise RuntimeError('checkpoint config.json is not a Qwen3-VL checkpoint')
        if args.dataset_id not in registered:
            raise RuntimeError(f'dataset {args.dataset_id!r} is not registered')
    if args.execute and (not args.smoke) and (args.dataset_id == 'mllm_demo'):
        raise RuntimeError('demo dataset is not a full training dataset')
    env['FORCE_TORCHRUN'] = '1'
    env['NPROC_PER_NODE'] = str(NPROC[args.model])
    env['PYTHONPATH'] = os.pathsep.join((part for part in (str(source / 'src'), env.get('PYTHONPATH', '')) if part))
    config = 'qwen3vl8b_lora_npu.yaml' if args.smoke else 'qwen3vl8b_drivelm_npu.yaml'
    command = ['llamafactory-cli', 'train', str(ROOT / config), f'model_name_or_path={args.model_path.resolve()}', f'dataset={args.dataset_id}', f'output_dir={output}']
    if protocol is not None:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=' + str(NPROC[args.model]), str(ROOT / 'tools/drivelm_npu_entry.py')] + command[2:]
        command += [f'dataset_dir={dataset_base}', f'eval_dataset={protocol["eval_dataset"]}',
                    f'cutoff_len={protocol["cutoff_len"]}']
    if args.smoke:
        command += ['max_steps=10', 'gradient_accumulation_steps=1', 'logging_steps=1', 'save_steps=10']
    cwd = source
    report = {'model': args.model, 'revision': LOCK[args.model]['revision'], 'patch_sha256': hashlib.sha256(patch_bytes(patch)).hexdigest() if patch else None, 'compat_patch_sha256': hashlib.sha256(patch_bytes(compat)).hexdigest() if compat else None, 'cwd': str(cwd), 'devices': args.devices, 'command': command, 'dataset_version': None, 'missing_data': missing, 'execute': args.execute}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if missing:
        raise RuntimeError('required data or weight files are missing')
    from fetch_qwen import verify
    verify(args.model_path.resolve())
    check = [sys.executable, str(ROOT / 'tools/check_env.py'), '--model', args.model, '--nproc', str(NPROC[args.model]), '--output', str(output / 'environment.json')]
    subprocess.run(check, env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    if protocol is not None:
        (output / 'full_data_verified.json').write_text(json.dumps(protocol, indent=2) + '\n')
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode == 0 and args.smoke:
        subprocess.run([sys.executable, str(ROOT / 'tools/check_qwen_checkpoint.py'), '--output-dir', str(output)], env=env, check=True)
    return completed.returncode
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(f'run failed: {exc}')
