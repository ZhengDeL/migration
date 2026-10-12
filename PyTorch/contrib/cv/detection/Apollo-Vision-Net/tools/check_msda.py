#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Compare mx_driving deformable attention forward/backward with PyTorch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch_npu
import mx_driving
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atol', type=float, default=0.03)
    parser.add_argument('--rtol', type=float, default=0.03)
    parser.add_argument('--output', type=Path, help='optional local JSON evidence file')
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise SystemExit('no NPU available')
    torch.manual_seed(20261010)
    device = 'npu:0'
    spatial = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
    starts = torch.tensor([0, 16], dtype=torch.long)
    value = torch.randn(1, 20, 8, 32, dtype=torch.float32, requires_grad=True)
    locations = torch.rand(1, 5, 8, 2, 4, 2, dtype=torch.float32, requires_grad=True)
    weights = torch.rand(1, 5, 8, 2, 4, dtype=torch.float32, requires_grad=True)
    cpu_out = multi_scale_deformable_attn_pytorch(value, spatial, locations, weights)
    cpu_out.sum().backward()
    npu_value = value.detach().to(device).requires_grad_()
    npu_locations = locations.detach().to(device).requires_grad_()
    npu_weights = weights.detach().to(device).requires_grad_()
    npu_out = mx_driving.multi_scale_deformable_attn(npu_value, spatial.to(device), starts.to(device), npu_locations, npu_weights)
    npu_out.sum().backward()
    torch.npu.synchronize()
    pairs = {'forward': (cpu_out, npu_out), 'grad_value': (value.grad, npu_value.grad), 'grad_locations': (locations.grad, npu_locations.grad), 'grad_weights': (weights.grad, npu_weights.grad)}
    report = {}
    for (name, (cpu, npu)) in pairs.items():
        assert cpu is not None and npu is not None
        npu_cpu = npu.detach().cpu()
        report[name] = {'max_abs_error': float((cpu.detach() - npu_cpu).abs().max()), 'allclose': bool(torch.allclose(cpu.detach(), npu_cpu, atol=args.atol, rtol=args.rtol)), 'finite': bool(torch.isfinite(npu_cpu).all())}
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + '\n', encoding='utf-8')
    print(result)
    return 0 if all((item['allclose'] and item['finite'] for item in report.values())) else 2
if __name__ == '__main__':
    raise SystemExit(main())
