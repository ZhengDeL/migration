#!/usr/bin/env python3
"""Download and validate the pinned public Qwen3-VL-8B-Instruct snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare import LOCK, ROOT


SPEC = LOCK["qwen3vl8b"]
ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json",
          "preprocessor_config.json", "video_preprocessor_config.json",
          "model.safetensors.index.json")


def verify(directory: Path) -> None:
    for name in ASSETS:
        if not (directory / name).is_file():
            raise RuntimeError(f"missing model asset: {directory / name}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise RuntimeError("downloaded checkpoint is not Qwen3-VL")
    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight map")
    for shard in set(weight_map.values()):
        path = directory / shard
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"model shard missing or incomplete: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "sources/Qwen3-VL-8B-Instruct")
    parser.add_argument("--execute", action="store_true", help="download the model snapshot")
    args = parser.parse_args()
    directory = args.destination.resolve()
    report = {"repo_id": SPEC["checkpoint"],
              "revision": SPEC["checkpoint_revision"],
              "destination": str(directory), "execute": args.execute}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=SPEC["checkpoint"],
                      revision=SPEC["checkpoint_revision"],
                      local_dir=str(directory), max_workers=4)
    verify(directory)
    (directory / ".source_revision").write_text(
        SPEC["checkpoint_revision"] + "\n", encoding="utf-8")
    print("validated pinned Qwen3-VL-8B checkpoint")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Qwen fetch failed: {exc}")
