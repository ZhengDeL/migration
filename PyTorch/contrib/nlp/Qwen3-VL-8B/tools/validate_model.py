#!/usr/bin/env python3
"""Validate this model's latest checkpoint and its own independent inference path."""
import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'qwen3vl8b'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--saved-only', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT):
        raise RuntimeError('结果必须在本模型目录中')
    report = {'model': MODEL, 'result': 'fail', 'official_accuracy_evaluation': False}
    try:
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch.npu.set_device(0)
        from check_qwen_checkpoint import check
        report['saved'] = check(output, 10, 2)
        if not args.saved_only:
            import subprocess
            subprocess.run([sys.executable, str(ROOT / 'tools/check_qwen_inference.py'), '--model-dir', str(ROOT / 'weights/qwen3-vl-8b-instruct'), '--adapter-dir', str(output)], check=True)
            report['inference'] = json.loads((output / 'inference_check.json').read_text())
        report['result'] = 'pass_saved_state' if args.saved_only else 'pass_independent_functional_validation'
    except Exception as error:
        report['failure'] = str(error)
        raise
    finally:
        (output / ('saved_state_check.json' if args.saved_only else 'validation_result.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
