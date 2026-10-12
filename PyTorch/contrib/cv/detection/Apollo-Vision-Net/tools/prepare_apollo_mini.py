#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Prepare real Apollo mini labels and images without changing their contents."""
import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path, PurePosixPath
import numpy as np
from world_mini_data import digest, require_hash
ANNOTATIONS = {'train': '8ac6738146470365807260e5533d1bb31ba5ceb6c01d7257302230269a491e2a', 'val': 'f231b25802bde99d59be63d0a9980f525bf740db9932bf41d679e81cbbf180df'}

def prepare(args):
    mini = {v['token'] for v in json.loads((args.nuscenes / 'v1.0-mini/sample.json').read_text())}
    prepared = args.prepared_dir.resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    (selected, records) = ({}, {})
    splits = ('val',) if args.val_only else ('train', 'val')
    for split in splits:
        annotation = args.data_dir / f'nuscenes_infos_temporal_{split}_occ_gt.pkl'
        require_hash(annotation, ANNOTATIONS[split])
        with annotation.open('rb') as stream:
            full = pickle.load(stream)
        infos = [copy.deepcopy(info) for info in full['infos'] if info['token'] in mini]
        if not infos:
            raise RuntimeError('No official Apollo annotations overlap nuScenes mini')
        for info in infos:
            name = PurePosixPath(info['occ_gt_path'])
            if '..' in name.parts or 'occ_gt_release_v1_0' not in name.parts:
                raise RuntimeError('Unexpected official label path')
            relative = PurePosixPath(*name.parts[name.parts.index('occ_gt_release_v1_0') + 1:])
            label = args.occ_dir / relative
            if not label.resolve().is_relative_to(args.occ_dir) or label.is_symlink():
                raise RuntimeError('Label must be a regular file inside OCC directory')
            labels = np.load(label, allow_pickle=False)
            if labels.ndim != 2 or labels.shape[1] != 2 or labels.size == 0 or (not np.isfinite(labels).all()) or (not np.equal(labels, np.floor(labels)).all()) or (labels[:, 0].min() < 0) or (labels[:, 0].max() >= 640000) or (labels[:, 1].min() < 0) or (labels[:, 1].max() > 15):
                raise RuntimeError('Invalid original sparse OCC label: ' + str(label))
            destination = prepared / 'occ_gt_release_v1_0' / relative
            if not destination.resolve().is_relative_to(prepared):
                raise RuntimeError('Label destination escapes prepared directory')
            destination.parent.mkdir(parents=True, exist_ok=True)
            checksum = digest(label)
            if label.resolve() != destination.resolve():
                shutil.copyfile(label, destination)
            require_hash(destination, checksum)
            records[str(destination.relative_to(prepared))] = {'path': str(destination.relative_to(prepared)), 'sha256': checksum, 'bytes': destination.stat().st_size, 'shape': list(labels.shape)}
            info['occ_gt_path'] = str(destination)
            for camera in info['cams'].values():
                name = PurePosixPath(camera['data_path'])
                if '..' in name.parts or 'nuscenes' not in name.parts:
                    raise RuntimeError('Unexpected official camera path')
                path = args.nuscenes / PurePosixPath(*name.parts[name.parts.index('nuscenes') + 1:])
                if not path.is_file():
                    raise RuntimeError('Missing nuScenes mini image: ' + str(path))
                camera['data_path'] = str(path)
            if len(info['cams']) != 6 or not np.isfinite(np.asarray(info['can_bus'])).all():
                raise RuntimeError('Incomplete camera or CAN bus annotation')
        metadata = copy.deepcopy(full['metadata'])
        metadata['version'] = 'v1.0-mini'
        selected[split] = {'infos': infos, 'metadata': metadata}
    split_disjoint = None
    if not args.val_only:
        train_tokens = {info['token'] for info in selected['train']['infos']}
        val_tokens = {info['token'] for info in selected['val']['infos']}
        if train_tokens & val_tokens or train_tokens | val_tokens != mini or len(train_tokens) != len(selected['train']['infos']) or (len(val_tokens) != len(selected['val']['infos'])):
            raise RuntimeError('Official mini train/val splits overlap, contain duplicates, or omit samples')
        split_disjoint = True
    if args.val_only:
        selected['train'] = copy.deepcopy(selected['val'])
        scope = f"{len(selected['train']['infos'])} official Apollo VAL frames overlapping nuScenes mini reused for functional training only; no heldout metric"
    else:
        scope = 'Official Apollo train/val split intersected with nuScenes mini; functional training only'
    files = {}
    for (split, value) in selected.items():
        value['metadata']['migration_scope'] = scope
        path = prepared / f'apollo_mini_{split}.pkl'
        with path.open('wb') as stream:
            pickle.dump(value, stream, protocol=4)
        files[path.name] = digest(path)
    archive_report = None
    archive_stamp = args.occ_dir / '.migration_occ_manifest.json'
    if archive_stamp.is_file():
        archive_report = json.loads(archive_stamp.read_text(encoding='utf-8'))
        expected = json.loads((Path(__file__).resolve().parents[1] / 'sources.lock.json').read_text(encoding='utf-8'))['apollo_visionnet']
        if archive_report.get('archive_sha256_observed') != expected['occ_archive_sha256_observed'] or not archive_report.get('gzip_crc_checked') or (not archive_report.get('tar_members_checked')):
            raise RuntimeError('OCC archive verification stamp does not match the locked release')
    report = {'model': 'apollo_visionnet', 'dataset_scope': scope, 'train_samples': len(selected['train']['infos']), 'heldout_samples': 0 if args.val_only else len(selected['val']['infos']), 'label_files': len(records), 'labels': list(records.values()), 'label_payloads_unchanged': True, 'prepared_files': files, 'annotation_observed_checksums': {name: ANNOTATIONS[name] for name in splits}, 'occupancy_source': 'extracted directory; selected regular files checked individually', 'full_archive_checked': archive_report is not None, 'archive_verification': archive_report, 'train_val_disjoint': split_disjoint, 'mini_samples_expected': len(mini), 'can_bus_zero_samples': sum((not np.asarray(v['can_bus']).any() for v in selected['train']['infos'])), 'initialization': 'verified separately when the training configuration is generated'}
    (prepared / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for (k, v) in report.items() if k != 'labels'}, indent=2), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--occ-dir', required=True, type=Path, help='extracted occ_gt_release_v1_0 directory')
    parser.add_argument('--archive', type=Path, help='verify and safely extract this full archive if OCC directory is absent')
    parser.add_argument('--nuscenes', required=True, type=Path)
    parser.add_argument('--prepared-dir', required=True, type=Path)
    parser.add_argument('--val-only', action='store_true', help='reuse original VAL samples for functional training, with no heldout evaluation')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve(strict=True)
    if not args.occ_dir.exists() and args.archive:
        from extract_apollo_occ import extract
        extract(args.archive, args.occ_dir)
    args.occ_dir = args.occ_dir.resolve(strict=True)
    args.nuscenes = args.nuscenes.resolve(strict=True)
    prepare(args)
