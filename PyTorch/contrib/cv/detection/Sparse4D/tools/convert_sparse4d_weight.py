#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Re-serialize the pinned Sparse4D ResNet50 weight for PyTorch 2.6+.

The official source uses legacy tar serialization, which PyTorch cannot read
with ``weights_only=True``. Run --execute in a container without network or NPU
devices and expose only the checkpoint directory. Never use this for an
unverified checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from prepare import LOCK

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(4 * 1024 * 1024)):
            digest.update(block)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if source == output or output.exists() or (not output.parent.is_dir()):
        parser.error('output must be a new file in an existing directory')
    source_hash = sha256(source)
    expected = LOCK['sparse4d']['pretrain_sha256']
    if source_hash != expected:
        raise RuntimeError(f'source SHA256 mismatch: expected {expected}, got {source_hash}')
    report = {'source': str(source), 'source_sha256': source_hash, 'official_url': LOCK['sparse4d']['pretrain_url'], 'output': str(output), 'execute': args.execute}
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    import torch
    weights = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(weights, dict) or len(weights) < 100:
        raise RuntimeError('expected a ResNet50 state_dict with at least 100 entries')
    if any((not isinstance(key, str) or not isinstance(value, torch.Tensor) for (key, value) in weights.items())):
        raise RuntimeError('checkpoint contains non-tensor state_dict entries')
    temporary = output.with_name(output.name + '.tmp')
    if temporary.exists():
        raise RuntimeError(f'temporary output already exists: {temporary}')
    try:
        torch.save(dict(weights), temporary)
        safe = torch.load(temporary, map_location='cpu', weights_only=True)
        if set(safe) != set(weights) or any((not torch.equal(weights[key], safe[key]) for key in weights)):
            raise RuntimeError('converted checkpoint differs from source tensors')
        output_hash = sha256(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report.update({'output_sha256': output_hash, 'tensor_count': len(weights), 'size': output.stat().st_size})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f'conversion failed: {exc}')
