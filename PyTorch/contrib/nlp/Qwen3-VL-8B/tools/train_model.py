#!/usr/bin/env python3
"""Train this model through its pinned upstream entry; save an independent result."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'qwen3vl8b'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['functional', 'full'], required=True)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--dataset_id', default='mllm_demo')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.mode == 'full':
        from prepare_full_data import require_full_protocol
        require_full_protocol(args.data_path, args.dataset_id)
    source = ROOT / 'sources' / PROFILE['source']
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise RuntimeError('Output escapes this model directory')
    smoke = args.mode == 'functional'
    devices = ','.join((str(i) for i in range(PROFILE['nproc'])))
    command = [sys.executable, str(ROOT / 'tools/run_train.py'), MODEL, '--source', str(source), '--devices', devices, '--output-dir', str(output)]
    command += ['--model-path', str(ROOT / 'weights/qwen3-vl-8b-instruct'), '--dataset-id', args.dataset_id]
    if smoke:
        command += ['--smoke']
    if not args.plan:
        command += ['--execute']
    env = os.environ.copy()
    env['ASCEND_RT_VISIBLE_DEVICES'] = devices
    env['PYTHONPATH'] = str(source / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode or args.plan:
        return result.returncode
    if smoke:
        subprocess.run([sys.executable, str(ROOT / 'tools/validate_model.py'), '--output_dir', str(output), '--saved-only'], check=True, env=env)
    report = {'model': MODEL, 'result': 'pass_functional_training' if smoke else 'training_completed', 'mode': args.mode, 'nproc': PROFILE['nproc'], 'upstream_revision': json.loads((ROOT / 'sources.lock.json').read_text())[MODEL]['revision'], 'entry_command': command, 'functional_accuracy_claimed': False}
    (output / 'train_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
