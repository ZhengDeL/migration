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
MODEL = 'sparse4d'

def require(path):
    if not path.exists():
        raise RuntimeError('完整训练资产缺失：' + str(path))
    return path

def full_metadata(raw):
    scenes = json.loads(require(raw / 'v1.0-trainval/scene.json').read_text())
    samples = json.loads(require(raw / 'v1.0-trainval/sample.json').read_text())
    if len(scenes) != 850 or len(samples) != 34149:
        raise RuntimeError('Expected complete official trainval: 850 scenes / 34149 samples')
    if len({x['token'] for x in scenes}) != 850 or len({x['token'] for x in samples}) != 34149:
        raise RuntimeError('Duplicate official full metadata tokens')
    return scenes, samples


def verify_annotations(paths, raw):
    scenes, samples = full_metadata(raw)
    from nuscenes.utils import splits
    scene_names = {x['token']: x['name'] for x in scenes}
    scene_for_sample = {x['token']: x['scene_token'] for x in samples}
    token_sets, records = [], []
    for path, split, expected in zip(paths, ('train', 'val'), (28130, 6019)):
        with require(path).open('rb') as stream:
            document = pickle.load(stream)
        infos = document['infos']
        tokens = {info['token'] for info in infos}
        allowed = set(getattr(splits, split))
        expected_tokens = {token for token, scene in scene_for_sample.items() if scene_names[scene] in allowed}
        if document.get('metadata', {}).get('version') != 'v1.0-trainval':
            raise RuntimeError('Full annotations must use original v1.0-trainval metadata')
        if len(infos) != expected or len(tokens) != expected or tokens != expected_tokens:
            raise RuntimeError('Original full split sample/token membership mismatch: ' + split)
        if any(len(info['cams']) != 6 for info in infos):
            raise RuntimeError('Every original sample must contain all six cameras')
        token_sets.append(tokens)
        records.append(dict(split=split, frames=expected,
            original_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if token_sets[0] & token_sets[1] or token_sets[0] | token_sets[1] != set(scene_for_sample):
        raise RuntimeError('Full train/val token overlap or incomplete union')
    return records


def annotation(original, target, raw):
    with original.open('rb') as stream:
        document = pickle.load(stream)
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
    return {'file': target.name, 'samples': len(document['infos']), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}

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
    full_metadata(raw)
    names = ('nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl')
    from prepare import require_checkout, patch_for, verify_applied_patch
    require_checkout(MODEL, source)
    verify_applied_patch(source, patch_for(MODEL, ROOT / PROFILE['sdk_dir']))
    conversion = None
    if not any((raw / name).exists() for name in names):
        from cpu_data_converter import load_author_converter
        converter, loading = load_author_converter(source / 'tools/nuscenes_converter.py')
        with tempfile.TemporaryDirectory(prefix='original_sparse_conversion_', dir=raw) as folder:
            candidate = Path(folder)
            converter.create_nuscenes_infos(str(raw), str(candidate / 'nuscenes'),
                                            version='v1.0-trainval', max_sweeps=10)
            verify_annotations([candidate / name for name in names], raw)
            for name in names:
                os.replace(candidate / name, raw / name)
        conversion = dict(implementation='pinned upstream create_nuscenes_infos', loading=loading)
    elif not all((raw / name).is_file() for name in names):
        raise RuntimeError('Preserve partial PKLs; require both complete trainval annotations')
    verified_splits = verify_annotations([raw / name for name in names], raw)
    import numpy as np
    anchor = raw / 'nuscenes_kmeans900.npy'
    anchor_receipt_path = raw / 'full_anchor_provenance.json'
    if not anchor.exists() and not anchor_receipt_path.exists():
        from cpu_data_converter import load_author_anchor_generator
        generator, loading = load_author_anchor_generator(source / 'tools/anchor_generator.py',
            source / 'projects/mmdet3d_plugin/core/box3d.py')
        with tempfile.TemporaryDirectory(prefix='original_sparse_anchors_', dir=raw) as folder:
            candidate = Path(folder) / anchor.name
            np.random.seed(42)
            generator.get_kmeans_anchor(str(raw / names[0]), num_anchor=900,
                detection_range=55, output_file_name=str(candidate))
            array = np.load(candidate, allow_pickle=False)
            if array.shape != (900, 11) or not np.isfinite(array).all():
                raise RuntimeError('Invalid original full train-only anchors')
            provenance = dict(input_split='v1.0-trainval/train', input_frames=28130,
                train_annotation_sha256=verified_splits[0]['original_sha256'],
                seed=42, num_anchor=900, detection_range=55, loading=loading,
                anchor_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
            os.replace(candidate, anchor)
            anchor_receipt_path.write_text(json.dumps(provenance, indent=2) + '\n')
    elif not anchor.is_file() or not anchor_receipt_path.is_file():
        raise RuntimeError('Preserve unproven/partial anchors; use a fresh complete-data directory or restore the matching full train-only provenance receipt')
    provenance = json.loads(anchor_receipt_path.read_text())
    array = np.load(anchor, allow_pickle=False)
    if (provenance['train_annotation_sha256'] != verified_splits[0]['original_sha256']
            or provenance['anchor_sha256'] != hashlib.sha256(anchor.read_bytes()).hexdigest()
            or provenance['input_split'] != 'v1.0-trainval/train' or provenance['input_frames'] != 28130
            or array.shape != (900, 11) or not np.isfinite(array).all()):
        raise RuntimeError('Anchor provenance/shape/numerics mismatch; mini anchors cannot replace full train-only generation')
    prepared = ROOT / 'prepared/full_nuscenes'
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-trainval'):
        link(raw / name, prepared / name)
    records = [annotation(data / name, prepared / name, raw) for name in names]
    link(prepared, source / 'data/nuscenes')
    link(prepared, source / 'data/nuscenes_anno_pkls')
    shutil.copyfile(anchor, source / 'nuscenes_kmeans900_full.npy')
    (prepared / 'preparation.json').write_text(json.dumps({'model': MODEL, 'dataset': 'v1.0-trainval', 'records': records,
        'verified_splits': verified_splits, 'original_conversion': conversion, 'anchor_provenance': provenance,
        'accuracy_verified': False}, indent=2) + '\n')
    print(json.dumps({'model': MODEL, 'result': 'complete_data_prepared', 'data_path': str(data), 'formal_training_verified': False}, indent=2))
if __name__ == '__main__':
    main()
