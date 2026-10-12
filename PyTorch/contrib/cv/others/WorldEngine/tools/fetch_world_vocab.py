#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Fetch only the pinned official WorldEngine trajectory vocabulary, with SHA256."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from prepare import LOCK, require_checkout

def verified(path: Path) -> bool:
    spec = LOCK['worldengine']
    return path.is_file() and path.stat().st_size == spec['vocab_bytes'] and (hashlib.sha256(path.read_bytes()).hexdigest() == spec['vocab_sha256'])

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    require_checkout('worldengine', source)
    spec = LOCK['worldengine']
    destination = source / spec['vocab_path']
    url = f"{spec['dataset_url']}/resolve/{spec['dataset_revision']}/{spec['vocab_path']}"
    print(json.dumps({'url': url, 'destination': str(destination), 'sha256': spec['vocab_sha256'], 'bytes': spec['vocab_bytes'], 'execute': args.execute}, indent=2))
    if not args.execute:
        return 0
    if verified(destination):
        print('verified: existing trajectory vocabulary')
        return 0
    import requests
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.npy.part')
    with requests.get(url, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    if not verified(temporary):
        raise RuntimeError('Trajectory vocabulary size or SHA256 does not match the pinned official dataset')
    temporary.replace(destination)
    print('verified: trajectory vocabulary only; images, annotations and checkpoint must be prepared separately')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
