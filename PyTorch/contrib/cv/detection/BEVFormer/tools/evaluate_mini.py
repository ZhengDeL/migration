#!/usr/bin/env python3
"""Strictly reload a pinned checkpoint and evaluate every mini validation frame.

Author trained weights and the legacy short-training checkpoint are separate profiles.
The upstream dataset formatter and nuScenes detection evaluator calculate metrics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'bevformer'
ANN = 'nuscenes_infos_temporal_val.pkl'
CHECKPOINT = 'epoch_1.pth'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--checkpoint-profile', choices=('delivered-mini', 'official-r101-24ep'),
        default='official-r101-24ep')
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'mini_evaluation.json').exists():
        raise RuntimeError('Use a fresh output directory inside this model project')
    started = time.monotonic()
    report = dict(model=MODEL, result='fail',
        collected_utc=datetime.now(timezone.utc).isoformat(),
        dataset_scope='nuScenes v1.0-mini, complete official mini_val split',
        expected_train_frames=323, expected_validation_frames=81,
        formal_trainval_accuracy_acceptance=False,
        quality_threshold_applied=False,
        limitation='Unconverged mini functional-training checkpoint; no trainval or GPU comparison',
        frames=[])
    try:
        import mmcv
        import numpy as np
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        from mmcv import Config
        from mmcv.device.npu import NPUDataParallel
        from mmcv.runner import wrap_fp16_model
        from mmdet.models import build_detector
        from mmdet.datasets import build_dataset, build_dataloader

        def finite(value):
            if torch.is_tensor(value):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            if hasattr(value, 'tensor'):
                return finite(value.tensor)
            return True

        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu(item) for item in value)
            if hasattr(value, 'tensor'):
                return value.to('cpu')
            return value

        torch.npu.set_device(0)
        reference = ROOT / 'validation/reference_run'
        if args.checkpoint_profile == 'official-r101-24ep':
            weight = json.loads((ROOT / 'evaluation.weights.lock.json').read_text())[args.checkpoint_profile]
            checkpoint = (ROOT / weight['path']).resolve(strict=True)
            if not checkpoint.is_relative_to(ROOT) or checkpoint.stat().st_size != weight['bytes']:
                raise RuntimeError('Author checkpoint path/size mismatch')
            expected = weight['sha256']
        else:
            checkpoint = reference / CHECKPOINT
            expected = json.loads((reference / 'validation_result.json').read_text())['saved']['sha256']
        actual = digest(checkpoint)
        if actual != expected:
            raise RuntimeError('Reference checkpoint SHA256 mismatch')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not finite(saved['state_dict']) or not finite(saved.get('optimizer', {})):
            raise RuntimeError('Non-finite checkpoint model/optimizer')
        if args.checkpoint_profile == 'official-r101-24ep':
            config_path = ROOT / 'sources' / PROFILE['source'] / weight['config']
            config_text = config_path.read_text()
            config = Config.fromfile(str(config_path))
            report.update(initialization='Original author released BEVFormer-base R101-DCN 24 epoch weights',
                limitation='Author trained checkpoint on mini_val; local full trainval training and GPU parity remain unverified')
        else:
            config_text = saved['meta']['config']
            config = Config.fromstring(config_text, '.py')
        report.update(checkpoint_sha256=actual,
            checkpoint=checkpoint.relative_to(ROOT).as_posix(),
            saved_config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
            checkpoint_profile=args.checkpoint_profile,
            checkpoint_updates=saved.get('meta', {}).get('iter') if args.checkpoint_profile == 'official-r101-24ep' else 162,
            checkpoint_reload='strict',
            torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
        source = ROOT / 'sources' / PROFILE['source']
        from prepare import require_checkout
        require_checkout(MODEL, source)
        os.chdir(source)
        sys.path.insert(0, str(source))
        importlib.import_module('projects.mmdet3d_plugin')
        config.model.pretrained = None
        test = config.data.val.copy()
        test.test_mode = True
        test.pop('samples_per_gpu', None)
        if MODEL == 'sparse4d':
            test.tracking = False  # Evaluate detection only, as in the original detection entrypoint.
        annotation = ROOT / 'prepared/nuscenes' / ANN
        test.ann_file = str(annotation)
        if 'data_root' in test:
            test.data_root = str(ROOT / 'data/nuscenes')
        train_annotation = annotation.with_name(ANN.replace('_val.pkl', '_train.pkl'))
        train = mmcv.load(str(train_annotation))
        validation = mmcv.load(str(annotation))
        train_tokens = {item['token'] for item in train['infos']}
        val_tokens = {item['token'] for item in validation['infos']}
        if (len(train_tokens), len(val_tokens)) != (323, 81) or train_tokens & val_tokens:
            raise RuntimeError('Mini train/validation coverage or disjointness mismatch')
        if validation['metadata']['version'] != 'v1.0-mini':
            raise RuntimeError('This entrypoint is restricted to mini')
        report.update(train_frames=323, validation_frames=81, train_val_disjoint=True,
            annotation_sha256=digest(annotation), train_annotation_sha256=digest(train_annotation),
            detection_only=True, validation_tokens=sorted(val_tokens))
        dataset = build_dataset(test)
        if len(dataset) != 81 or {x['token'] for x in dataset.data_infos} != val_tokens:
            raise RuntimeError('Loader does not cover the complete mini validation split')
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
            num_gpus=1, dist=False, shuffle=False)
        model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        model.load_state_dict(saved['state_dict'], strict=True)
        report['model_tensors_loaded'] = len(saved['state_dict'])
        del saved
        model = model.to('npu:0').eval()
        if config.get('fp16') is not None:
            wrap_fp16_model(model)
        report['precision'] = 'upstream-config FP16 wrap' if config.get('fp16') is not None else 'FP32'
        wrapped = NPUDataParallel(model, device_ids=[0])
        results = []
        with torch.no_grad():
            for index, batch in enumerate(loader):
                prediction = wrapped(return_loss=False, rescale=True, **batch)
                if not isinstance(prediction, list) or len(prediction) != 1 or not finite(prediction):
                    raise RuntimeError('Invalid/non-finite frame prediction at ' + str(index))
                results.extend(cpu(prediction))
                token = dataset.data_infos[index]['token']
                report['frames'].append(dict(index=index, token=token, outputs_finite=True))
                if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                    print('mini inference: %d/%d frames' % (index + 1, len(dataset)), flush=True)
        torch.npu.synchronize()
        if len(results) != 81:
            raise RuntimeError('Incomplete validation outputs')
        raw = output / 'predictions.pkl'
        mmcv.dump(results, str(raw))
        report['predictions_sha256'] = digest(raw)
        metrics = dataset.evaluate(results, jsonfile_prefix=str(output / 'official_detection'))
        metrics = {key: float(value) for key, value in metrics.items()}
        primary = {key: value for key, value in metrics.items() if key.endswith(('/NDS', '/mAP'))}
        if len(primary) != 2 or not all(np.isfinite(value) for value in primary.values()):
            raise RuntimeError('Missing/non-finite original mAP/NDS')
        undefined = [key for key, value in metrics.items() if not np.isfinite(value)]
        allowed_undefined = ('traffic_cone_orient_err', 'traffic_cone_vel_err',
            'traffic_cone_attr_err', 'barrier_vel_err', 'barrier_attr_err')
        if any(not key.endswith(allowed_undefined) for key in undefined):
            raise RuntimeError('Unexpected undefined metric fields: ' + repr(undefined))
        report['undefined_metric_fields'] = undefined
        report['undefined_metric_encoding'] = 'null: nuScenes does not define these class-specific orientation/velocity/attribute metrics'
        report['metrics'] = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
        report['official_evaluator_on_mini'] = True
        report['evaluator'] = 'Original source dataset.evaluate / nuScenes DetectionEval, mini_val'
        report['frames_checked'] = len(results)
        report['metric_files'] = [dict(path=p.relative_to(output).as_posix(),
            sha256=digest(p), bytes=p.stat().st_size) for p in sorted(output.rglob('*.json'))]
        report['result'] = 'pass_complete_mini_detection_evaluation'
    except Exception as error:
        report['failure'] = '%s: %s' % (type(error).__name__, error)
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        report['timing_scope'] = 'Concurrent validation wall time; not an isolated performance benchmark'
        (output / 'mini_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('frames', 'validation_tokens')},
            indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
