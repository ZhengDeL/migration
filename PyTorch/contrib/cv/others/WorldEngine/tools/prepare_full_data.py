#!/usr/bin/env python3
"""Validate and connect this model's complete official training dataset."""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from prepare_data import link, rewrite
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'worldengine'

def require(path):
    if not path.exists():
        raise RuntimeError('完整训练资产缺失：' + str(path))
    return path

def annotation(original, target, raw):
    require(original)
    with original.open('rb') as stream:
        document = pickle.load(stream)
    if document.get('metadata', {}).get('version') != 'v1.0-trainval':
        raise RuntimeError('完整训练标注必须为 v1.0-trainval：' + str(original))
    document = rewrite(document, raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        pickle.dump(document, stream, pickle.HIGHEST_PROTOCOL)
    return {'file': target.name, 'samples': len(document['infos']), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_path.resolve(strict=True)
    if not data.is_relative_to(ROOT):
        raise RuntimeError('完整数据必须属于当前模型目录')
    source = ROOT / 'sources' / PROFILE['source']
    for name in ('raw/openscene-v1.1/sensor_blobs/trainval', 'raw/openscene-v1.1/sensor_blobs/test', 'alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl', 'alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl', 'alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl', 'alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl', 'alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth'):
        require(data / name)
    link(data / 'raw', source / 'data/raw')
    for name in ('merged_infos_navformer', 'pdms_cache', 'ckpts'):
        link(data / 'alg_engine' / name, source / 'data/alg_engine' / name)
    print(json.dumps({'model': MODEL, 'result': 'complete_data_prepared', 'data_path': str(data), 'formal_training_verified': False}, indent=2))
if __name__ == '__main__':
    main()
