#!/usr/bin/env python3
"""Prepare source/data links inside this single-model directory."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'apollo_visionnet'

def link(target, source):
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        if source.resolve() == target.resolve():
            return
        source.unlink()
    elif source.exists():
        raise RuntimeError('保留已有真实目录，拒绝覆盖：' + str(source))
    source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=target.is_dir())

def rewrite(value, data):
    if isinstance(value, dict):
        return {key: rebase(item, data) if key == 'data_path' and isinstance(item, str) else rewrite(item, data) for (key, item) in value.items()}
    if isinstance(value, list):
        return [rewrite(item, data) for item in value]
    if isinstance(value, tuple):
        return tuple((rewrite(item, data) for item in value))
    return value

def rebase(name, data):
    pieces = Path(name).parts
    for anchor in ('samples', 'sweeps'):
        if anchor in pieces:
            result = data / Path(*pieces[pieces.index(anchor):])
            if not result.is_file():
                raise RuntimeError('真实图像缺失：' + str(result))
            return str(result)
    return name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_path.resolve(strict=True)
    source = ROOT / 'sources' / PROFILE['source']
    prep = [sys.executable, str(ROOT / 'tools/prepare.py'), MODEL, '--source', str(source), '--apply']
    subprocess.run(prep, check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/prepare_apollo_mini.py'), '--data-dir', str(data), '--occ-dir', str(data / 'occ_gt_release_v1_0'), '--nuscenes', str(ROOT / 'data/nuscenes'), '--prepared-dir', str(ROOT / 'prepared/apollo'), *(['--archive', str(data / 'occ_gt_release_v1_0.tar.gz')] if not (data / 'occ_gt_release_v1_0').is_dir() else [])], check=True)
    print(json.dumps({'model': MODEL, 'result': 'pass_single_model_preparation', 'data_path': str(data)}, indent=2), flush=True)
if __name__ == '__main__':
    main()
