#!/usr/bin/env python3
"""Small two-rank NPU backward and HCCL smoke; not a model acceptance test."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.npu.device_count() < 2:
        raise RuntimeError("this check requires exactly two visible NPU ranks")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("HCCL rank mapping mismatch")
        torch.manual_seed(20261010)
        model = torch.nn.Linear(4, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4) / 32
        x = x + local_rank * 0.25
        target = x.sum(dim=1, keepdim=True)
        before = torch.nn.functional.mse_loss(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        before.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("gradient missing or non-finite")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)
        weight_before = model.weight.detach().clone()
        optimizer.step()
        after = torch.nn.functional.mse_loss(model(x), target)
        marker = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        dist.all_reduce(marker, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        changed = bool(torch.any(model.weight.detach() != weight_before))
        finite = bool(torch.isfinite(before) and torch.isfinite(after))
        collective = float(marker.item()) == 3.0
        report = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": torch.npu.get_device_name(local_rank),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "backend": dist.get_backend(),
            "loss_before": float(before.item()),
            "loss_after": float(after.item()),
            "weight_changed": changed,
            "finite": finite,
            "all_reduce_sum": float(marker.item()),
            "pass": bool(changed and finite and collective),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{rank}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        dist.barrier()
        return 0 if report["pass"] else 2
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
