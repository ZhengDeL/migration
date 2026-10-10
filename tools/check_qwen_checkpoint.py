#!/usr/bin/env python3
"""Check the saved artifacts of the two-rank Qwen LoRA smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check(directory: Path, steps: int, ranks: int) -> dict:
    import torch
    from safetensors.torch import load_file

    state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    checkpoint = directory / f"checkpoint-{steps}"
    saved_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != steps or saved_state.get("global_step") != steps:
        raise RuntimeError(f"expected completed trainer step {steps}")
    history = [row for row in state.get("log_history", []) if "loss" in row]
    if {row.get("step") for row in history} != set(range(1, steps + 1)):
        raise RuntimeError("per-step loss history is incomplete")
    for row in history:
        if not all(math.isfinite(float(row[key])) for key in ("loss", "grad_norm")):
            raise RuntimeError(f"non-finite loss or gradient at step {row.get('step')}")

    adapter = directory / "adapter_model.safetensors"
    adapter_hash = sha256_file(adapter)
    if adapter_hash != sha256_file(checkpoint / "adapter_model.safetensors"):
        raise RuntimeError("final adapter differs from the last checkpoint")
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("init_lora_weights") is not True:
        raise RuntimeError("smoke update check requires default zero-initialized LoRA B")
    tensors = load_file(str(adapter), device="cpu")
    if not tensors or not all(torch.isfinite(value).all().item() for value in tensors.values()):
        raise RuntimeError("adapter is empty or has non-finite tensors")
    b_weights = {key: value for key, value in tensors.items() if ".lora_B." in key}
    nonzero_b = sum(bool(torch.count_nonzero(value).item()) for value in b_weights.values())
    if not b_weights or nonzero_b == 0:
        raise RuntimeError("no LoRA B update was found")

    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer_state = optimizer.get("state", {})
    if not optimizer_state:
        raise RuntimeError("optimizer state is empty")
    optimizer_steps = sorted({int(value["step"]) for value in optimizer_state.values()})
    if optimizer_steps != [steps]:
        raise RuntimeError(f"unexpected optimizer steps: {optimizer_steps}")
    for value in optimizer_state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if not torch.isfinite(value[key]).all().item():
                raise RuntimeError(f"optimizer has non-finite {key}")
    rank_files = [checkpoint / f"rng_state_{rank}.pth" for rank in range(ranks)]
    if not all(path.is_file() for path in rank_files):
        raise RuntimeError("a distributed rank RNG artifact is missing")
    return {"result": "pass_functional", "output_dir": str(directory),
            "global_step": steps, "rank_artifacts": ranks,
            "loss_first": history[0]["loss"], "loss_last": history[-1]["loss"],
            "loss_min": min(row["loss"] for row in history),
            "loss_max": max(row["loss"] for row in history),
            "adapter_bytes": adapter.stat().st_size, "adapter_sha256": adapter_hash,
            "adapter_tensors": len(tensors), "lora_b_tensors": len(b_weights),
            "nonzero_lora_b_tensors": nonzero_b,
            "optimizer_states": len(optimizer_state), "optimizer_steps": optimizer_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.ranks < 1:
        parser.error("steps and ranks must be positive")
    directory = args.output_dir.resolve(strict=True)
    report = check(directory, args.steps, args.ranks)
    (directory / "checkpoint_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"checkpoint check failed: {exc}")
