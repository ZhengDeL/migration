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
MODEL = 'bevformer'

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
        checkpoint = output / 'epoch_1.pth'
        expected = 162

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
        if not loss_records:
            raise RuntimeError('缺少本次实际训练的 loss 日志')
        report['saved'] = {'checkpoint': checkpoint.name, 'sha256': digest(checkpoint), 'model_tensors': len(states), 'optimizer_states': len(optimizer['state']), 'optimizer_steps': steps, 'loss_records': len(loss_records), 'meta_iter': saved.get('meta', {}).get('iter'), 'result': 'pass_finite_saved_state'}
        if not args.saved_only:
            from mmcv import Config
            from mmcv.device.npu import NPUDataParallel
            from mmcv.runner import wrap_fp16_model
            from mmdet.models import build_detector
            from mmdet.datasets import build_dataset, build_dataloader
            source = ROOT / 'sources' / PROFILE['source']
            os.chdir(source)
            sys.path.insert(0, str(source))
            importlib.import_module('projects.mmdet3d_plugin')
            config = Config.fromstring(saved['meta']['config'], '.py')
            config.model.pretrained = None
            test = config.data.val.copy()
            test.test_mode = True
            test.pop('samples_per_gpu', None)
            ann_name = 'nuscenes_infos_temporal_val.pkl'
            test.ann_file = str(ROOT / 'prepared/nuscenes' / ann_name)
            if 'data_root' in test:
                test.data_root = str(ROOT / 'data/nuscenes')
            dataset = build_dataset(test)
            loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, shuffle=False)
            model = build_detector(config.model, test_cfg=config.get('test_cfg'))
            model.load_state_dict(states, strict=True)
            model = model.to('npu:0').eval()
            if config.get('fp16') is not None:
                wrap_fp16_model(model)
            wrapped = NPUDataParallel(model, device_ids=[0])
            with torch.no_grad():
                prediction = wrapped(return_loss=False, rescale=True, **next(iter(loader)))
            if not prediction or not finite(prediction):
                raise RuntimeError('独立验证帧推理为空或非有限')
            report['inference'] = {'result': 'pass_single_frame_functional_inference', 'frames_checked': 1, 'checkpoint_reload': 'strict', 'outputs_finite': True, 'official_metrics': False}
        report['result'] = 'pass_saved_state' if args.saved_only else 'pass_independent_functional_validation'
    except Exception as error:
        report['failure'] = str(error)
        raise
    finally:
        (output / ('saved_state_check.json' if args.saved_only else 'validation_result.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
