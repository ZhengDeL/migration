#!/usr/bin/env python3
"""Validate and connect this model's complete official training dataset."""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from prepare_data import link, rebase
from full_annotation_checks import audit, label_path, NAMES, LABEL_FIELDS
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'apollo_visionnet'

def require(path):
    if not path.exists():
        raise RuntimeError('完整训练资产缺失：' + str(path))
    return path

def annotation(original, target, raw, occ):
    require(original)
    with original.open('rb') as stream:
        document = pickle.load(stream)
    if document.get('metadata', {}).get('version') != 'v1.0-trainval':
        raise RuntimeError('完整训练标注必须为 v1.0-trainval：' + str(original))
    def rewrite_paths(value):
        if isinstance(value, dict):
            return {key: rebase(item, raw) if key in ('data_path', 'lidar_path') and isinstance(item, str)
                    else str(label_path(item, occ)) if key in LABEL_FIELDS and isinstance(item, str)
                    else rewrite_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite_paths(item) for item in value]
        if isinstance(value, tuple):
            return tuple(rewrite_paths(item) for item in value)
        return value
    document = rewrite_paths(document)
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
    raw = ROOT / 'data/nuscenes_full'
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval/scene.json'):
        require(raw / name)
    canbus = require(ROOT / 'data/can_bus')
    if not any(canbus.glob('scene-*_pose.json')):
        raise RuntimeError('完整训练需要官方 CAN bus scene-*_pose.json，拒绝 mini 的空目录')
    occ = require(data / 'occ_gt_release_v1_0')
    for name in ('train', 'val', 'occ_gt_train.json', 'occ_gt_val.json'):
        require(occ / name)
    verified = audit([data / name for name in NAMES], raw, occ, check_media=True)
    prepared = ROOT / 'prepared/full_nuscenes'
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval'):
        link(raw / name, prepared / name)
    names = NAMES
    records = [annotation(data / name, prepared / name, raw, occ) for name in names]
    link(prepared, source / 'data/nuscenes')
    link(canbus, source / 'data/can_bus')
    occ_prepared = ROOT / 'prepared/full_occ'
    occ_prepared.mkdir(parents=True, exist_ok=True)
    for name in ('train', 'val', 'occ_gt_train.json', 'occ_gt_val.json'):
        link(occ / name, occ_prepared / name)
    for name in names:
        link(prepared / name, occ_prepared / name)
    link(occ_prepared, source / 'data/occ_gt_release_v1_0')
    (prepared / 'preparation.json').write_text(json.dumps({'model': MODEL, 'dataset': 'v1.0-trainval', 'records': records,
        'original_full_annotation_audit': verified, 'accuracy_verified': False}, indent=2) + '\n')
    print(json.dumps({'model': MODEL, 'result': 'complete_data_prepared', 'data_path': str(data), 'formal_training_verified': False}, indent=2))
if __name__ == '__main__':
    main()
