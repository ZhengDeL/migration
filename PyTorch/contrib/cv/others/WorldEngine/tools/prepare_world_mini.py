#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Prepare a pinned real NAVTEST subset for WorldEngine functional training."""
import argparse
import copy
import json
import pickle
import shutil
import tarfile
from pathlib import Path, PurePosixPath
import yaml
from prepare import require_checkout
from world_mini_data import ASSETS, LOGS, SCOPE, digest, require_hash

def extract_selected(archive_path, root, wanted=None):
    root.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError('Unsafe archive path')
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError('Unexpected archive link or special file')
            relative = name.as_posix().removeprefix('./')
            if wanted is not None and relative not in wanted:
                continue
            destination = root / relative
            if not destination.resolve().is_relative_to(root.resolve()):
                raise RuntimeError('Extraction escapes its root')
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.extractfile(member)
            with destination.open('wb') as output:
                shutil.copyfileobj(payload, output, 8 * 1024 * 1024)
            if destination.stat().st_size != member.size:
                raise RuntimeError('Truncated archive member')
            extracted.add(relative)
    if wanted is not None and extracted != wanted:
        raise RuntimeError('Missing labels: ' + repr(sorted(wanted - extracted)[:6]))
    return extracted

def dump_pickle(path, value):
    with path.open('wb') as output:
        pickle.dump(value, output, protocol=4)

def prepare_world(args):
    directory = args.data_dir
    selection = {'logs': LOGS, 'dataset_scope': SCOPE, 'camera': {'path': ASSETS[-1][2], 'lfs': {'oid': ASSETS[-1][4]}}}
    annotation = directory / 'nuplan_openscene_navtest.pkl'
    cache = directory / 'pdm_8192_gt_cache_navtest.pkl'
    camera_archive = directory / Path(selection['camera']['path']).name
    checksums = {'annotation': require_hash(annotation, '59a8bdf1a9a2c6066dd562333679f54ffee9cbfe9cb3420bab27dce62c5da21b'), 'pdm_cache': require_hash(cache, '58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316'), 'camera': require_hash(camera_archive, selection['camera']['lfs']['oid'])}
    prepared = directory / 'prepared'
    cameras = prepared / 'cameras'
    files = extract_selected(camera_archive, cameras)
    by_name = {}
    for relative in files:
        name = Path(relative).name
        by_name.setdefault(name, []).append(cameras / relative)
    with annotation.open('rb') as stream:
        full = pickle.load(stream)
    infos = full['infos'] if isinstance(full, dict) else full
    logs = set(selection['logs'])
    infos = [copy.deepcopy(info) for info in infos if info['log_name'] in logs]
    if not infos:
        raise RuntimeError('Selected camera logs absent from official NAVTEST annotations')
    for info in infos:
        if len(info['cams']) != 8:
            raise RuntimeError('Expected eight real WorldEngine camera views')
        for camera in info['cams'].values():
            matches = by_name.get(Path(camera['data_path']).name, [])
            if len(matches) != 1:
                raise RuntimeError('Missing or ambiguous official camera: ' + camera['data_path'])
            camera['data_path'] = str(matches[0])
    split = yaml.safe_load((args.source / 'projects/AlgEngine/configs/navsim_splits/navtest_split/navtest.yaml').read_text())
    allowed = set(split.get('tokens', split.get('scenario_tokens', [])))
    with cache.open('rb') as stream:
        pdm = pickle.load(stream)
    anchors = []
    for (index, info) in enumerate(infos):
        if info['token'] not in allowed or info['token'] not in pdm:
            continue
        if index < 3 or index + 8 >= len(infos):
            continue
        if any((v['log_token'] != info['log_token'] for v in infos[index - 3:index + 9])):
            continue
        anchors.append(info['token'])
    anchors = anchors[:64]
    if len(anchors) < 4:
        raise RuntimeError('Too few complete official temporal anchors')
    dump_pickle(prepared / 'worldengine_navtest_smoke.pkl', {'infos': infos, 'metadata': {'migration_scope': selection['dataset_scope'], 'source_logs': sorted(logs)}})
    dump_pickle(prepared / 'worldengine_navtest_smoke_pdm.pkl', {token: pdm[token] for token in anchors})
    subset_split = copy.deepcopy(split)
    subset_split.pop('scenario_tokens', None)
    subset_split['tokens'] = anchors
    subset_split['log_names'] = sorted(logs)
    (prepared / 'worldengine_navtest_smoke.yaml').write_text(yaml.safe_dump(subset_split))
    report = {'model': 'worldengine', 'dataset_scope': selection['dataset_scope'], 'logs': sorted(logs), 'annotation_frames': len(infos), 'training_anchors': len(anchors), 'camera_files': len(files), 'pdm_tokens': len(anchors), 'publisher_checksums': checksums, 'trajectory_labels_changed': False, 'diffusiondrive_data_mode': False, 'heldout_samples': 0, 'prepared_files': {name: digest(prepared / name) for name in ('worldengine_navtest_smoke.pkl', 'worldengine_navtest_smoke_pdm.pkl', 'worldengine_navtest_smoke.yaml')}}
    (prepared / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    args.data_dir = args.data_dir.resolve(strict=True)
    require_checkout('worldengine', args.source)
    prepare_world(args)
