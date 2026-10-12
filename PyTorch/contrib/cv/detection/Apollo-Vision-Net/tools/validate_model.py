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
MODEL = 'apollo_visionnet'

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
        from world_mini_data import digest
        checkpoint = output / 'iter_10.pth'
        expected = 10

        def finite(value):
            if torch.is_tensor(value):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all((finite(item) for item in value.values()))
            if isinstance(value, (tuple, list)):
                return all((finite(item) for item in value))
            if hasattr(value, 'tensor'):
                return finite(value.tensor)
            return True
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        states = saved.get('state_dict', {})
        optimizer = saved.get('optimizer', {})
        if not states or not optimizer or (not finite(states)) or (not finite(optimizer)):
            raise RuntimeError('保存模型/优化器为空或包含非有限值')
        steps = sorted({float(item['step']) for item in optimizer.get('state', {}).values() if 'step' in item})
        if steps != [float(expected)]:
            raise RuntimeError('优化器更新次数不符合本次功能训练范围：' + repr(steps))
        loss_records = []
        for path in output.glob('*.log.json'):
            for line in path.read_text().splitlines():
                item = json.loads(line)
                if item.get('mode') == 'train' and 'loss' in item:
                    if not math.isfinite(float(item['loss'])):
                        raise RuntimeError('训练 loss 非有限')
                    loss_records.append(item)
        report['saved'] = {'checkpoint': checkpoint.name, 'sha256': digest(checkpoint), 'model_tensors': len(states), 'optimizer_states': len(optimizer['state']), 'optimizer_steps': steps, 'loss_records': len(loss_records), 'meta_iter': saved.get('meta', {}).get('iter'), 'result': 'pass_finite_saved_state'}
        ranks = [json.loads((output / f'training_check_rank{i}.json').read_text()) for i in range(4)]
        if any((r.get('result') != 'pass_training_smoke' or r.get('iterations') != 10 or r.get('parameter_sync') != 'pass' or (not r.get('parameter_changed')) for r in ranks)):
            raise RuntimeError('四个 rank 的训练/参数更新检查不完整')
        if ranks[0]['checkpoint_sha256'] != report['saved']['sha256']:
            raise RuntimeError('保存文件 SHA 与本次 rank 0 记录不一致')
        report['ranks_checked'] = 4
        if not args.saved_only:
            del saved, states, optimizer
            import subprocess
            subprocess.run([sys.executable, str(ROOT / 'tools/check_apollo_heldout.py'), '--source', str(ROOT / 'sources/Apollo-Vision-Net'), '--data-dir', str(ROOT / 'data/apollo'), '--prepared-dir', str(ROOT / 'prepared/apollo'), '--output-dir', str(output)], check=True)
            report['inference'] = json.loads((output / 'heldout_check.json').read_text())
        report['result'] = 'pass_saved_state' if args.saved_only else 'pass_independent_functional_validation'
    except Exception as error:
        report['failure'] = str(error)
        raise
    finally:
        (output / ('saved_state_check.json' if args.saved_only else 'validation_result.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
