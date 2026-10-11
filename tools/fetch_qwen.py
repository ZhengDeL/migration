#!/usr/bin/env python3
"""Download and validate the pinned public Qwen3-VL-8B-Instruct snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare import LOCK, ROOT


SPEC = LOCK["qwen3vl8b"]
ASSETS = ("config.json", "generation_config.json", "chat_template.json",
          "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
          "preprocessor_config.json", "video_preprocessor_config.json",
          "model.safetensors.index.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    shards = set(weight_map.values())
    expected = SPEC["checkpoint_shards_sha256"]
    if shards != set(expected):
        raise RuntimeError("model shard list differs from pinned snapshot")
    for shard in sorted(shards):
        path = directory / shard
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            raise RuntimeError(f"model shard missing or incomplete: {path}")
        digest = sha256_file(path)
        if digest != expected[shard]:
            raise RuntimeError(f"model shard SHA256 mismatch: {shard}")
        print(f"verified {shard}: sha256:{digest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "sources/Qwen3-VL-8B-Instruct")
    parser.add_argument("--hub", choices=("huggingface", "modelscope"),
                        default="huggingface", help="official publisher mirror; both verify pinned weight hashes")
    parser.add_argument("--execute", action="store_true", help="download the model snapshot")
    args = parser.parse_args()
    directory = args.destination.resolve()
    repo_id = SPEC["checkpoint"] if args.hub == "huggingface" else SPEC["checkpoint_modelscope"]
    revision = (SPEC["checkpoint_revision"] if args.hub == "huggingface"
                else SPEC["checkpoint_modelscope_revision"])
    report = {"hub": args.hub, "repo_id": repo_id,
              "revision": revision,
              "destination": str(directory), "execute": args.execute}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if args.hub == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, revision=revision,
                          local_dir=str(directory), max_workers=4)
    else:
        from modelscope import snapshot_download

        needed = [name for name in ASSETS if not (directory / name).is_file()]
        for name, expected in SPEC["checkpoint_shards_sha256"].items():
            path = directory / name
            if not path.is_file() or path.stat().st_size < 1024 * 1024 or sha256_file(path) != expected:
                needed.append(name)
        if needed:
            snapshot_download(model_id=repo_id, revision=revision,
                              local_dir=str(directory), allow_file_pattern=needed,
                              max_workers=4)
    verify(directory)
    marker = ".source_revision" if args.hub == "huggingface" else ".source_modelscope_revision"
    (directory / marker).write_text(revision + "\n", encoding="utf-8")
    print(f"validated pinned Qwen3-VL-8B checkpoint from {args.hub}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Qwen fetch failed: {exc}")
