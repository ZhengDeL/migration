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
MODEL = 'bevformer'

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
    prep += ['--sdk', str(ROOT / PROFILE['sdk_dir'])]
    subprocess.run(prep, check=True)
    raw = ROOT / 'data/annotations_original'
    manifest = json.loads((raw / 'manifest.json').read_text())
    prepared = ROOT / 'prepared/nuscenes'
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-mini'):
        link(data / name, prepared / name)
    records = []
    for (name, expected) in manifest['files'].items():
        path = raw / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('原标注 SHA 不一致：' + name)
        with path.open('rb') as stream:
            document = pickle.load(stream)
        document = rewrite(document, data)
        if document.get('metadata', {}).get('version') != 'v1.0-mini':
            raise RuntimeError('本离线功能数据必须为 v1.0-mini，正式数据另外准备。')
        target = prepared / name
        with target.open('wb') as stream:
            pickle.dump(document, stream, pickle.HIGHEST_PROTOCOL)
        records.append({'file': str(target), 'samples': len(document['infos'])})
    source_data = source / 'data'
    link(prepared, source_data / 'nuscenes')
    canbus = ROOT / 'data/can_bus'
    canbus.mkdir(parents=True, exist_ok=True)
    link(canbus, source_data / 'can_bus')
    (prepared / 'preparation.json').write_text(json.dumps({'model': MODEL, 'records': records, 'dataset': 'v1.0-mini', 'images_rebased_to': str(data)}, indent=2) + '\n')
    print(json.dumps({'model': MODEL, 'result': 'pass_single_model_preparation', 'data_path': str(data)}, indent=2), flush=True)
if __name__ == '__main__':
    main()
