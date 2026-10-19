#!/usr/bin/env python3
"""Validate and connect this model's complete official training dataset."""
import argparse
import hashlib
import json
import pickle
import shutil
import os
import tempfile
from pathlib import Path
from prepare_data import link, rebase
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text())
MODEL = 'bevformer'

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
    def rewrite_media(value):
        if isinstance(value, dict):
            return {key: rebase(item, raw) if key in ('data_path', 'lidar_path') and isinstance(item, str)
                    else rewrite_media(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite_media(item) for item in value]
        if isinstance(value, tuple):
            return tuple(rewrite_media(item) for item in value)
        return value
    document = rewrite_media(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as stream:
        pickle.dump(document, stream, pickle.HIGHEST_PROTOCOL)
    return {'file': target.name, 'samples': len(document['infos']),
            'frames_with_nonzero_can': sum(any(float(x) != 0 for x in info['can_bus']) for info in document['infos']),
            'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_path.resolve(strict=True)
    if not data.is_relative_to(ROOT):
        raise RuntimeError('完整数据必须属于当前模型目录')
    source = ROOT / 'sources' / PROFILE['source']
    raw = data
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval/scene.json'):
        require(raw / name)
    canbus = require(ROOT / 'data/can_bus')
    if not any(canbus.glob('scene-*_pose.json')):
        raise RuntimeError('完整训练需要官方 CAN bus scene-*_pose.json，拒绝 mini 的空目录')
    scenes = json.loads((raw / 'v1.0-trainval/scene.json').read_text())
    samples = json.loads(require(raw / 'v1.0-trainval/sample.json').read_text())
    if len(scenes) != 850 or len(samples) != 34149:
        raise RuntimeError('Expected complete official trainval metadata: 850 scenes / 34149 samples')
    names = ('nuscenes_infos_temporal_train.pkl', 'nuscenes_infos_temporal_val.pkl')
    conversion = None
    if not any((raw / name).exists() for name in names):
        from prepare import require_checkout, patch_for, compat_patch_for, verify_applied_patch
        require_checkout(MODEL, source)
        patch, compat = patch_for(MODEL, ROOT / PROFILE['sdk_dir']), compat_patch_for(MODEL)
        verify_applied_patch(source, patch, (compat,) if compat else ())
        # Keep the author's conversion functions; isolate model-only imports.
        from cpu_data_converter import load_author_converter
        converter = source / 'tools/data_converter/nuscenes_converter.py'
        module, conversion_loading = load_author_converter(converter)
        with tempfile.TemporaryDirectory(prefix='official_temporal_conversion_', dir=raw) as folder:
            candidate = Path(folder)
            module.create_nuscenes_infos(str(raw), str(candidate), str(canbus.parent),
                'nuscenes', version='v1.0-trainval', max_sweeps=10)
            counts = []
            for name, expected_count in zip(names, (28130, 6019)):
                with require(candidate / name).open('rb') as stream:
                    document = pickle.load(stream)
                if document['metadata']['version'] != 'v1.0-trainval' or len(document['infos']) != expected_count:
                    raise RuntimeError('Original full temporal conversion produced incomplete annotations')
                counts.append(expected_count)
            for name in names:
                os.replace(candidate / name, raw / name)
            conversion = dict(implementation='pinned upstream create_nuscenes_infos',
                converter_sha256=hashlib.sha256(converter.read_bytes()).hexdigest(),
                train_frames=counts[0], val_frames=counts[1], loading=conversion_loading)
    elif not all((raw / name).is_file() for name in names):
        raise RuntimeError('Preserve partial annotation files; provide both complete trainval PKLs')
    prepared = ROOT / 'prepared/full_nuscenes'
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval'):
        link(raw / name, prepared / name)
    records = [annotation(data / name, prepared / name, raw) for name in names]
    if [record['samples'] for record in records] != [28130, 6019]:
        raise RuntimeError('Expected complete original train/val frame counts')
    if not sum(record['frames_with_nonzero_can'] for record in records):
        raise RuntimeError('All complete-data CAN vectors are zero; verify the official CAN root')
    token_sets = []
    for name, expected_count in zip(names, (28130, 6019)):
        with (prepared / name).open('rb') as stream:
            infos = pickle.load(stream)['infos']
        tokens = {info['token'] for info in infos}
        if len(tokens) != expected_count:
            raise RuntimeError('Duplicate full-data sample tokens')
        token_sets.append(tokens)
    if token_sets[0] & token_sets[1] or token_sets[0] | token_sets[1] != {sample['token'] for sample in samples}:
        raise RuntimeError('Full train/val token coverage or disjointness mismatch')
    link(prepared, source / 'data/nuscenes')
    link(canbus, source / 'data/can_bus')
    (prepared / 'preparation.json').write_text(json.dumps({'model': MODEL, 'dataset': 'v1.0-trainval', 'records': records, 'original_conversion': conversion, 'accuracy_verified': False}, indent=2) + '\n')
    print(json.dumps({'model': MODEL, 'result': 'complete_data_prepared', 'data_path': str(data), 'formal_training_verified': False}, indent=2))
if __name__ == '__main__':
    main()
