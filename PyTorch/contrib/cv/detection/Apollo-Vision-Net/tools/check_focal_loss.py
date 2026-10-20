#!/usr/bin/env python3
"""Compare adapted NPU focal losses and gradients with the original CPU formula.

Inputs come from one real training batch and a completed local mini checkpoint.
The probe stops after capturing both detection and occupancy focal-loss inputs.
It does not update model parameters or establish complete model accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if not output.is_relative_to(ROOT) or not checkpoint.is_relative_to(ROOT):
        raise RuntimeError('Use this independent model directory for checkpoint and output')
    if (output / 'focal_loss_precision.json').exists():
        raise RuntimeError('Use a fresh output directory')
    report = dict(model='apollo_visionnet', result='fail', collected_utc=datetime.now(timezone.utc).isoformat(),
        scope='Actual detection/occupancy loss inputs from one original mini training batch',
        full_model_accuracy_accepted=False, optimizer_created=False, parameter_updates=0,
        seed=42, cases=[], tolerances=dict(loss_atol=1e-5, loss_rtol=1e-4,
            gradient_atol=1e-7, gradient_rtol=1e-3, gradient_relative_l2=1e-4,
            gradient_cosine_minimum=0.999999))
    try:
        from prepare import require_checkout, patch_for, compat_patch_for, verify_applied_patch
        from validate_legacy_mini import validate
        source = ROOT / 'sources/Apollo-Vision-Net'
        require_checkout('apollo_visionnet', source)
        verify_applied_patch(source, patch_for('apollo_visionnet', None),
                             (compat_patch_for('apollo_visionnet'),))
        manifest = validate('apollo_visionnet', ROOT / 'prepared/apollo', ROOT / 'data/apollo')
        if (manifest['train_samples'], manifest['heldout_samples']) != (242, 162):
            raise RuntimeError('Use the complete original OCC/mini training intersection')
        ranks = [json.loads((checkpoint.parent / f'quality_training_rank{rank}.json').read_text())
                 for rank in range(4)]
        if any(r['result'] != 'pass_completed_mini_training' or r['updates'] != 1464 for r in ranks):
            raise RuntimeError('Complete the four-rank mini training first')
        report['checkpoint_sha256'] = digest(checkpoint)
        if report['checkpoint_sha256'] != ranks[0]['checkpoint_sha256']:
            raise RuntimeError('Checkpoint SHA differs from the completed training receipt')
        import numpy as np
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        from mmcv import Config
        from mmcv.device.npu import NPUDataParallel
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.npu.manual_seed_all(42)
        torch.npu.set_device(0)
        os.chdir(source)
        sys.path.insert(0, str(source))
        importlib.import_module('projects.mmdet3d_plugin')
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader
        from projects.mmdet3d_plugin.models.npu_focal_loss import NpuFocalLoss
        from mmdet.models.losses.focal_loss import FocalLoss
        config_path = checkpoint.parent / 'mini_training.py'
        cfg = Config.fromfile(str(config_path), import_custom_modules=False)
        cfg.model.pretrained = None
        cfg.data.train.ann_file = str(ROOT / 'prepared/apollo/apollo_mini_train.pkl')
        cfg.data.train.data_root = str(ROOT / 'data/nuscenes')
        dataset = build_dataset(cfg.data.train)
        if len(dataset) != 242:
            raise RuntimeError('Original training split coverage changed')
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                 num_gpus=1, dist=False, shuffle=False)
        batch = next(iter(loader))
        def sample_tokens(value):
            if isinstance(value, dict):
                if isinstance(value.get('sample_idx'), str):
                    return [value['sample_idx']]
                return [token for child in value.values() for token in sample_tokens(child)]
            if isinstance(value, (list, tuple)):
                return [token for child in value for token in sample_tokens(child)]
            if type(value).__name__ == 'DataContainer':
                return sample_tokens(value.data)
            return []
        actual_tokens = sample_tokens(batch.get('img_metas'))
        if not actual_tokens:
            raise RuntimeError('Record actual sample tokens from the collated training metadata')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.load_state_dict(saved['state_dict'], strict=True)
        report['model_tensors_loaded'] = len(saved['state_dict'])
        del saved
        model = model.to('npu:0').eval()
        report.update(config_sha256=digest(config_path), torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__, training_frames=len(dataset),
            requested_dataset_index=0, annotation_token_at_requested_index=dataset.data_infos[0]['token'],
            actual_batch_sample_tokens=actual_tokens,
            original_cpu_focal_source_sha256=digest(inspect.getsourcefile(FocalLoss)),
            adapted_focal_source_sha256=digest(inspect.getsourcefile(NpuFocalLoss)))
        original = NpuFocalLoss.forward
        captured = set()

        class CapturedBothLosses(Exception):
            pass

        def observed(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
            classes = pred.shape[-1]
            if classes not in (10, 16) or classes in captured:
                return original(self, pred, target, weight, avg_factor, reduction_override)
            with torch.enable_grad():
                npu_pred = pred.detach().clone().requires_grad_(True)
                cpu_pred = pred.detach().cpu().requires_grad_(True)
                cpu_target = target.detach().cpu()
                cpu_weight = weight.detach().cpu() if torch.is_tensor(weight) else weight
                cpu_factor = avg_factor.detach().cpu() if torch.is_tensor(avg_factor) else avg_factor
                npu_loss = original(self, npu_pred, target.detach(), weight, avg_factor, reduction_override)
                cpu_loss = FocalLoss.forward(self, cpu_pred, cpu_target, cpu_weight,
                                            cpu_factor, reduction_override)
                if npu_loss.numel() != 1 or cpu_loss.numel() != 1:
                    raise RuntimeError('This real loss case must use its original scalar reduction')
                npu_grad = torch.autograd.grad(npu_loss, npu_pred)[0].detach().cpu()
                cpu_grad = torch.autograd.grad(cpu_loss, cpu_pred)[0].detach()
                actual_loss = npu_loss.detach().cpu()
            for value in (cpu_pred, cpu_loss, actual_loss, cpu_grad, npu_grad):
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError('Non-finite real loss input, output or gradient')
            difference = npu_grad.double() - cpu_grad.double()
            reference_norm = torch.linalg.vector_norm(cpu_grad.double())
            actual_norm = torch.linalg.vector_norm(npu_grad.double())
            if not reference_norm.item() or not actual_norm.item():
                raise RuntimeError('The real gradient comparison must be nonzero')
            relative_l2 = float(torch.linalg.vector_norm(difference) / reference_norm)
            cosine = float(torch.sum(npu_grad.double() * cpu_grad.double()) / (reference_norm * actual_norm))
            limits = report['tolerances']
            case = dict(name='occupancy' if classes == 16 else 'detection',
                pred_shape=list(pred.shape), target_shape=list(target.shape), pred_dtype=str(pred.dtype),
                background_target_count=int((cpu_target == classes).sum()),
                foreground_target_count=int((cpu_target < classes).sum()),
                original_loss_weight=self.loss_weight, original_gamma=self.gamma, original_alpha=self.alpha,
                reduction=reduction_override or self.reduction,
                avg_factor=float(cpu_factor) if cpu_factor is not None else None,
                input_sha256=hashlib.sha256(cpu_pred.detach().contiguous().numpy().tobytes()).hexdigest(),
                target_sha256=hashlib.sha256(cpu_target.contiguous().numpy().tobytes()).hexdigest(),
                cpu_loss=float(cpu_loss.detach()), npu_loss=float(actual_loss),
                loss_abs_difference=float((actual_loss - cpu_loss.detach()).abs()),
                gradient_elements=cpu_grad.numel(), gradient_max_abs_difference=float(difference.abs().max()),
                gradient_reference_l2=float(reference_norm), gradient_relative_l2=relative_l2,
                gradient_cosine=cosine,
                loss_allclose=bool(torch.allclose(actual_loss, cpu_loss.detach(),
                    atol=limits['loss_atol'], rtol=limits['loss_rtol'])),
                gradient_allclose=bool(torch.allclose(npu_grad, cpu_grad,
                    atol=limits['gradient_atol'], rtol=limits['gradient_rtol'])))
            case['passed'] = (case['loss_allclose'] and case['gradient_allclose']
                and relative_l2 <= limits['gradient_relative_l2'] and cosine >= limits['gradient_cosine_minimum'])
            report['cases'].append(case)
            print(json.dumps(case), flush=True)
            if not case['passed']:
                raise RuntimeError('Adapted focal loss/gradient differs from the original CPU formula')
            captured.add(classes)
            if captured == {10, 16}:
                raise CapturedBothLosses()
            return original(self, pred, target, weight, avg_factor, reduction_override)

        NpuFocalLoss.forward = observed
        try:
            with torch.no_grad():
                NPUDataParallel(model, device_ids=[0])(return_loss=True, **batch)
        except CapturedBothLosses:
            report['stopped_after_two_real_loss_inputs'] = True
        finally:
            NpuFocalLoss.forward = original
        if captured != {10, 16}:
            raise RuntimeError('Did not capture both original detection and occupancy loss inputs')
        report['result'] = 'pass_original_cpu_vs_adapted_npu_focal_loss_and_gradients'
    except Exception as error:
        report['failure'] = repr(error)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'focal_loss_precision.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
