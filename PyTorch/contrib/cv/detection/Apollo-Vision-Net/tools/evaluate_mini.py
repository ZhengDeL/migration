#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Reload the delivered Apollo checkpoint and run original occupancy metrics on every mini heldout frame."""
import argparse
import importlib
import json
import os
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from prepare import require_checkout, patch_for, compat_patch_for, patch_state, verify_applied_patch
from validate_legacy_mini import validate
from world_mini_data import digest, require_hash

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, help='A completed, verified model-local mini-training checkpoint')
    args = parser.parse_args()
    args.source = ROOT / 'sources/Apollo-Vision-Net'
    args.data_dir = ROOT / 'data/apollo'
    args.prepared_dir = ROOT / 'prepared/apollo'
    source = args.source.resolve(strict=True)
    prepared = args.prepared_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'mini_evaluation.json').exists():
        raise RuntimeError('Use a fresh output directory inside this model project')
    reference = ROOT / 'validation/reference_run'
    require_checkout('apollo_visionnet', source)
    patch = patch_for('apollo_visionnet', None)
    compat = compat_patch_for('apollo_visionnet')
    if patch_state(source, patch) != 'applied' or patch_state(source, compat) != 'applied':
        raise RuntimeError('Apply the locked Apollo patches first')
    verify_applied_patch(source, patch, (compat,))
    manifest = validate('apollo_visionnet', prepared, args.data_dir)
    if not manifest.get('heldout_samples') or manifest.get('train_val_disjoint') is not True:
        raise RuntimeError('Independent Apollo mini validation data is required; VAL reuse is not accepted')
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve(strict=True)
        if not checkpoint.is_relative_to(ROOT):
            raise RuntimeError('Checkpoint must belong to this independent model directory')
        reports = [json.loads((checkpoint.parent / f'quality_training_rank{i}.json').read_text()) for i in range(4)]
        if any(r.get('result') != 'pass_completed_mini_training' or r.get('updates') != 1464 for r in reports):
            raise RuntimeError('Complete the four-card 24 epoch mini training first')
        config = checkpoint.parent / 'mini_training.py'
        updates = 1464
        initialization = 'official DLA depth backbone followed by 24 complete mini train epochs'
    else:
        reports = [json.loads((reference / f'training_check_rank{i}.json').read_text()) for i in range(4)]
        if any((r.get('result') != 'pass_training_smoke' or r.get('iterations') != 10 for r in reports)):
            raise RuntimeError('Complete the four-card mini training first')
        if any((r.get('dataset_scope') != manifest['dataset_scope'] + '; official DLA depth pretrained backbone' for r in reports)):
            raise RuntimeError('Training result does not match the prepared dataset scope')
        checkpoint = reference / 'iter_10.pth'
        config = reference / 'apollo_visionnet_real_smoke.py'
        updates = 10
        initialization = 'official DLA depth pretrained backbone before ten-step mini training'
    if any((not r.get('initialization_tensor_matches') for r in reports)):
        raise RuntimeError('Official DLA initialization was not verified before training')
    require_hash(checkpoint, reports[0]['checkpoint_sha256'])
    report = {'model': 'apollo_visionnet', 'test': 'checkpoint_reload_heldout_inference', 'checkpoint_sha256': reports[0]['checkpoint_sha256'], 'config_sha256': digest(config), 'train_samples': manifest['train_samples'], 'heldout_samples': manifest['heldout_samples'], 'train_val_disjoint': True, 'official_accuracy_evaluation': False, 'precision': 'FP32', 'physical_cards': 1, 'initialization': 'official DLA depth pretrained backbone before ten-step mini training', 'frames': [], 'result': 'fail'}
    from datetime import datetime, timezone
    report.update(collected_utc=datetime.now(timezone.utc).isoformat(),
        dataset_scope=manifest['dataset_scope'], checkpoint_updates=updates, initialization=initialization,
        formal_trainval_accuracy_acceptance=False, quality_threshold_applied=False,
        detection_metrics_computed=False,
        detection_limitation='Original mini OCC heldout is 162 frames; original detection evaluator requires official mini_val (81). No split substitution.',
        evaluator='Original dataset.evaluate_occ_iou and SSCMetrics; no evaluator code changes')
    started = time.monotonic()
    try:
        import sys
        import numpy as np
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        from mmcv import Config
        from mmcv.device.npu import NPUDataParallel
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        os.chdir(source)
        sys.path.insert(0, str(source))
        importlib.import_module('projects.mmdet3d_plugin')
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader
        torch.npu.set_device(0)

        def finite(value):
            if torch.is_tensor(value):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, np.ndarray):
                return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
            if isinstance(value, dict):
                return all((finite(item) for item in value.values()))
            if isinstance(value, (list, tuple)):
                return all((finite(item) for item in value))
            if hasattr(value, 'tensor'):
                return finite(value.tensor)
            return True
        cfg = Config.fromfile(str(config), import_custom_modules=False)
        cfg.model.pretrained = None
        cfg.data.test.ann_file = str(prepared / 'apollo_mini_val.pkl')
        cfg.data.test.data_root = str(ROOT / 'data/nuscenes')
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)
        dataset = build_dataset(cfg.data.test)
        if len(dataset) != manifest['heldout_samples']:
            raise RuntimeError('Heldout dataset size differs from the checked manifest')
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1, dist=False, shuffle=False)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not finite(saved['state_dict']) or not finite(saved['optimizer']):
            raise RuntimeError('Non-finite saved model or optimizer state')
        model.load_state_dict(saved['state_dict'], strict=True)
        report['model_tensors_loaded'] = len(saved['state_dict'])
        del saved
        model = model.to('npu:0').eval()
        raw = []
        occupancy_predictions = []

        def check_head(module, inputs, outputs):
            if not finite(outputs):
                raise RuntimeError('Non-finite raw inference head output')
            if not isinstance(outputs, dict) or not torch.is_tensor(outputs.get('occupancy_preds')):
                raise RuntimeError('Apollo inference did not produce occupancy logits')
            occupancy = outputs['occupancy_preds']
            if occupancy.numel() != 640000 * 16:
                raise RuntimeError('Unexpected occupancy output size')
            raw.append({'occupancy_shape': list(occupancy.shape), 'finite': True})
        hook = model.pts_bbox_head.register_forward_hook(check_head)
        wrapped = NPUDataParallel(model, device_ids=[0])
        with torch.no_grad():
            for (index, batch) in enumerate(loader):
                prediction = wrapped(return_loss=False, rescale=True, **batch)
                if not isinstance(prediction, tuple) or len(prediction) != 2 or (not finite(prediction)):
                    raise RuntimeError('Invalid or non-finite decoded Apollo prediction')
                (boxes, occupancy) = prediction
                if not isinstance(occupancy, dict) or not torch.is_tensor(occupancy.get('occupancy_preds')):
                    raise RuntimeError('Original Apollo occupancy result contract mismatch')
                occupancy_predictions.append(occupancy['occupancy_preds'].detach().cpu())
                if not isinstance(boxes, list) or len(boxes) != 1 or (not occupancy):
                    raise RuntimeError('Apollo inference result is incomplete')
                bbox = boxes[0]['pts_bbox']
                if bbox['scores_3d'].numel() == 0:
                    raise RuntimeError('No decoded detection predictions')
                if len(raw) != index + 1:
                    raise RuntimeError('Raw head checks did not cover every validation frame')
                report['frames'].append({'token': dataset.data_infos[index]['token'], 'boxes': int(bbox['scores_3d'].numel()), 'raw_head': raw[-1], 'decoded_finite': True})
                if (index + 1) % 20 == 0:
                    print(f'Apollo heldout inference: {index + 1}/{len(dataset)} frames checked', flush=True)
        hook.remove()
        torch.npu.synchronize()
        if len(report['frames']) != manifest['heldout_samples']:
            raise RuntimeError('Not all heldout frames were checked')
        report['checkpoint_reload'] = 'pass_strict_model_state'
        report['frames_checked'] = len(report['frames'])
        if len(occupancy_predictions) != len(dataset):
            raise RuntimeError('Incomplete occupancy predictions')
        import mmcv
        raw_predictions = output / 'occupancy_predictions.pkl'
        mmcv.dump(occupancy_predictions, str(raw_predictions))
        report['predictions_sha256'] = digest(raw_predictions)
        dataset.evaluate_occ_iou(occupancy_predictions, flow_results=None)
        stats = dataset.eval_metrics.get_stats()
        report['metrics'] = {key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in stats.items()}
        if not finite({key: np.asarray(value) for key, value in report['metrics'].items()}):
            raise RuntimeError('Non-finite original occupancy metrics')
        report['metric_units'] = {'iou': 'percent', 'miou': 'percent',
            'iou_ssc': 'percent', 'precision': 'fraction', 'recall': 'fraction'}
        report['class_names'] = list(dataset.occupancy_class_names)
        report['confusion_matrix_reference_rows_prediction_columns'] = dataset.eval_metrics.hist.tolist()
        report['occupancy_evaluator_sha256'] = digest(source / 'projects/mmdet3d_plugin/datasets/occupancy_metrics.py')
        report['official_evaluator_on_mini'] = True
        report['result'] = 'pass_complete_mini_occupancy_evaluation'
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (output / 'mini_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: value for (key, value) in report.items() if key != 'frames'}, indent=2), flush=True)
if __name__ == '__main__':
    main()
