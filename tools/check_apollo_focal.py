#!/usr/bin/env python3
"""Compare Apollo's NPU focal loss and gradients with the upstream CPU formula."""

import argparse
import importlib.util
import json
from pathlib import Path

from prepare import (compat_patch_for, patch_for, patch_state, require_checkout,
                     verify_applied_patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    require_checkout('apollo_visionnet', source)
    patch = patch_for('apollo_visionnet', None)
    compat = compat_patch_for('apollo_visionnet')
    if patch_state(source, patch) != 'applied' or patch_state(source, compat) != 'applied':
        raise RuntimeError('Apply the pinned migration patches first')
    verify_applied_patch(source, patch, (compat,))
    import torch
    import torch_npu
    from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(20261011)
    path = source / 'projects/mmdet3d_plugin/models/npu_focal_loss.py'
    spec = importlib.util.spec_from_file_location('migration_apollo_focal', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = []
    for classes, loss_weight in ((10, 2.0), (16, 10.0)):
        for reduction, avg_factor in (('none', None), ('sum', None), ('mean', None), ('mean', 23.0)):
            logits = (torch.randn(257, classes) * 5).requires_grad_()
            targets = torch.arange(257, dtype=torch.long) % (classes + 1)
            weights = torch.linspace(0, 1, 257)
            one_hot = F.one_hot(targets, classes + 1)[:, :classes]
            reference = loss_weight * py_sigmoid_focal_loss(
                logits, one_hot, weights, gamma=2.0, alpha=0.25,
                reduction=reduction, avg_factor=avg_factor)
            reference.sum().backward()
            npu_logits = logits.detach().to('npu').requires_grad_()
            loss = module.NpuFocalLoss(loss_weight=loss_weight, reduction=reduction)(
                npu_logits, targets.to('npu'), weights.to('npu'), avg_factor=avg_factor)
            loss.sum().backward()
            actual, gradient = loss.detach().cpu(), npu_logits.grad.cpu()
            if not torch.allclose(reference.detach(), actual, atol=1e-5, rtol=1e-4):
                raise RuntimeError('NPU focal loss differs from the upstream CPU reference')
            if not torch.allclose(logits.grad, gradient, atol=1e-5, rtol=1e-4):
                raise RuntimeError('NPU focal gradient differs from the upstream CPU reference')
            reports.append({'classes': classes, 'reduction': reduction, 'avg_factor': avg_factor,
                            'max_loss_abs_error': float((reference.detach() - actual).abs().max()),
                            'max_gradient_abs_error': float((logits.grad - gradient).abs().max())})
    report = {'model': 'apollo_visionnet', 'test': 'focal_loss_cpu_npu_parity',
              'dtype': 'float32', 'cases': reports, 'atol': 1e-5, 'rtol': 1e-4, 'result': 'pass'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
