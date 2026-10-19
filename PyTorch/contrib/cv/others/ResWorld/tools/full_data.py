"""Strict model-local full-trainval data checks, without selecting an NPU."""
import json
from pathlib import Path
from common import ROOT, own_path, sha256

SCOPE = 'official_nuscenes_trainval_resworld_v1'
EXPECTED = {'train_frames': 28130, 'val_frames': 6019, 'train_scenes': 700, 'val_scenes': 150}


def inspect_raw(path):
    path = own_path(path, exists=True)
    table = path / 'v1.0-trainval'
    scenes = json.loads((table / 'scene.json').read_text())
    samples = json.loads((table / 'sample.json').read_text())
    if len(scenes) != 850 or len(samples) != 34149:
        raise RuntimeError('Full trainval requires exactly 850 scenes and 34,149 annotated samples.')
    if len({item['token'] for item in samples}) != len(samples):
        raise RuntimeError('Duplicate official sample tokens.')
    from nuscenes.utils.splits import train, val
    names = {item['name'] for item in scenes}
    if names != set(train) | set(val) or set(train) & set(val):
        raise RuntimeError('The full scene names differ from the official train/val split.')
    for name in ['maps/expansion/boston-seaport.json', 'maps/expansion/singapore-hollandvillage.json',
                 'maps/expansion/singapore-onenorth.json', 'maps/expansion/singapore-queenstown.json']:
        if not (path / name).is_file():
            raise RuntimeError('Missing full official map: ' + name)
    if not any((path / 'can_bus').glob('scene-*_pose.json')):
        raise RuntimeError('Original official CAN messages are required.')
    count = 0
    for item in json.loads((table / 'sample_data.json').read_text()):
        file = own_path(path / item['filename'], exists=True)
        if not file.is_relative_to(path) or not file.is_file() or not file.stat().st_size:
            raise RuntimeError('Missing/unsafe full sensor record: ' + item['filename'])
        count += 1
    labels = json.loads((table / 'lidarseg.json').read_text())
    if len(labels) != 34149:
        raise RuntimeError('Full ResWorld requires all 34,149 original lidarseg labels.')
    for item in labels:
        file = own_path(path / item['filename'], exists=True)
        if not file.is_relative_to(path) or not file.is_file():
            raise RuntimeError('Missing original lidarseg input.')
    return {'scenes': scenes, 'samples': samples, 'sensor_files': count, 'lidarseg_labels': len(labels)}


def check_prepared(path):
    path = own_path(path, exists=True)
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata.get('scope') != SCOPE or metadata.get('counts') != EXPECTED:
        raise RuntimeError('This is not a complete prepared ResWorld trainval dataset.')
    raw = inspect_raw(path)
    manifest = path / 'input_manifest.jsonl'
    if sha256(manifest) != metadata['manifest_sha256']:
        raise RuntimeError('Full input-manifest identity differs.')
    count = 0
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            file = own_path(path / row['path'], exists=True)
            if not file.is_relative_to(path) or file.stat().st_size != row['bytes'] or sha256(file) != row['sha256']:
                raise RuntimeError('Full input digest differs: ' + row['path'])
            count += 1
    if count != metadata['manifest_files'] or raw['sensor_files'] != metadata['sensor_files']:
        raise RuntimeError('Full input file coverage differs from the preparation.')
    return metadata
