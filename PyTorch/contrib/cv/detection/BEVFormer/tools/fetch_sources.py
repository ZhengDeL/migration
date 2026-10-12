#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Clone the exact upstream commits listed in sources.lock.json."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from prepare import LOCK, ROOT, require_checkout
DIRECTORIES = {'bevformer': 'BEVFormer'}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', choices=tuple(DIRECTORIES), help='defaults to this model upstream repository')
    parser.add_argument('--destination', type=Path, default=ROOT / 'sources')
    parser.add_argument('--execute', action='store_true', help='perform network downloads')
    args = parser.parse_args()
    destination = args.destination.resolve()
    names = args.models or list(DIRECTORIES)
    actions = []
    for name in names:
        source = destination / DIRECTORIES[name]
        if source.exists():
            require_checkout(name, source)
            state = 'existing_pinned_checkout'
        elif args.execute:
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', LOCK[name]['url'], str(source)], check=True)
            subprocess.run(['git', 'checkout', '--detach', LOCK[name]['revision']], cwd=source, check=True)
            require_checkout(name, source)
            state = 'downloaded_pinned_checkout'
        else:
            state = 'planned_download'
        actions.append({'model': name, 'revision': LOCK[name]['revision'], 'source': str(source), 'state': state})
    print(json.dumps(actions, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'fetch failed: {exc}')
