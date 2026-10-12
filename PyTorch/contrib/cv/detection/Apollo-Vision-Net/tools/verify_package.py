#!/usr/bin/env python3
"""Verify the immutable files of this model's delivery package."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets-only', action='store_true')
args = parser.parse_args()
manifest = ROOT / 'package_manifest.json'
if not manifest.is_file():
    raise SystemExit('仅源码仓不含离线资产；请取得本模型的完整交付包。')
data = json.loads(manifest.read_text())
profile = json.loads((ROOT / 'model.json').read_text())
if data.get('schema_version') != 1 or data.get('model') != profile['model'] or (not data.get('files')):
    raise RuntimeError('不是当前模型的有效交付清单')
names = [item['path'] for item in data['files']]
if len(set(names)) != len(names):
    raise RuntimeError('交付清单存在重复路径')
for item in data['files']:
    path = (ROOT / item['path']).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size != item['bytes'] or (len(item['sha256']) != 64):
        raise RuntimeError('文件缺失或路径/大小错误：' + item['path'])
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    if sha.hexdigest() != item['sha256']:
        raise RuntimeError('文件 SHA256 不一致：' + item['path'])
print('Single-model package verification passed:', data['model'], len(data['files']), 'files', flush=True)
