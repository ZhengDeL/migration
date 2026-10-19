"""Derive full ResWorld annotations/depth with the pinned author's own programs.

This prepares inputs only; it cannot mark training or task accuracy as passed.
"""
import argparse
import json
import os
from pathlib import Path
from common import ROOT, own_path, sha256, write_json
from full_data import SCOPE, EXPECTED, inspect_raw, check_prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', type=Path, default=ROOT / 'data/nuscenes_full')
    a = p.parse_args()
    data = own_path(a.data_path, exists=True)
    # Validate full raw identity before any output or training-source mutation.
    raw = inspect_raw(data)
    if (data / 'metadata.json').exists():
        metadata = check_prepared(data)
        print(json.dumps({'result': 'existing_complete_inputs_reverified', 'counts': metadata['counts']}))
        return
    import numpy as np
    import mmcv
    from prepare_data import load_module
    from prepare_source import prepare_source
    from cpu_data_converter import load_author_converter
    source = prepare_source()
    converter, converter_loading = load_author_converter(source / 'tools/data_converter/vad_nuscenes_converter.py')
    os.chdir(ROOT)
    progress = data / 'preparation_progress.json'
    write_json(progress, {'stage': 'original_full_annotation_converter', 'training_completed': False})
    names = ['vad_nuscenes_infos_temporal_' + split + '.pkl' for split in ['train', 'val']]
    existing = [(data / name).exists() for name in names]
    if any(existing) and not all(existing):
        raise RuntimeError('Keep the existing partial annotation attempt; prepare a fresh data directory.')
    annotations_generated = not all(existing)
    if annotations_generated:
        converter.create_nuscenes_infos(str(data), str(data), str(data), 'vad_nuscenes',
                                        version='v1.0-trainval', max_sweeps=10)
    scene_by_token = {row['token']: row for row in raw['scenes']}
    sample_by_token = {row['token']: row for row in raw['samples']}
    from nuscenes.utils.splits import train, val
    splits, annotations, counts, fallback = {}, {}, {}, {}
    all_tokens, expected_depth = set(), set()

    def normalize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in ['lidar_path', 'data_path'] and isinstance(item, str):
                    file = Path(item)
                    file = own_path(file if file.is_absolute() else ROOT / file, exists=True)
                    if not file.is_relative_to(data):
                        raise RuntimeError('Converted sensor path escapes this model dataset.')
                    result[key] = file.relative_to(ROOT).as_posix()
                else:
                    result[key] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalize(item) for item in value)
        return value

    for split, name, official_names in zip(['train', 'val'], names, [set(train), set(val)]):
        document = mmcv.load(str(data / name))
        if document.get('metadata', {}).get('version') != 'v1.0-trainval':
            raise RuntimeError('The original converter did not produce trainval metadata.')
        infos = document['infos']
        tokens = {row['token'] for row in infos}
        if len(infos) != EXPECTED[split + '_frames'] or len(tokens) != len(infos):
            raise RuntimeError('Full converted annotation counts/tokens differ: ' + split)
        scene_tokens = {row['scene_token'] for row in infos}
        if {scene_by_token[token]['name'] for token in scene_tokens} != official_names:
            raise RuntimeError('Converted annotation scene split differs: ' + split)
        for row in infos:
            if row['scene_token'] != sample_by_token[row['token']]['scene_token']:
                raise RuntimeError('Converted annotation disagrees with the official scene token.')
            if len(row['cams']) != 6:
                raise RuntimeError('All six official cameras are required.')
            for camera in row['cams'].values():
                image = Path(camera['data_path'])
                image = own_path(image if image.is_absolute() else ROOT / image, exists=True)
                relative = image.relative_to(data)
                if relative.parts[0] != 'samples':
                    raise RuntimeError('Expected official keyframe camera paths under samples/.')
                expected_depth.add((Path('samples_point_label') / Path(*relative.parts[1:])).with_suffix('.npy').as_posix())
        if tokens & all_tokens:
            raise RuntimeError('Training/validation sample overlap.')
        all_tokens |= tokens
        counts[split + '_frames'], counts[split + '_scenes'] = len(infos), len(scene_tokens)
        splits[split] = sorted(scene_tokens)
        fallback[split] = sum(int(not np.any(np.asarray(row['can_bus']))) for row in infos)
        converted = normalize(document)
        temporary = (data / name).with_suffix('.normalized.pkl')
        mmcv.dump(converted, str(temporary))
        temporary.replace(data / name)
        annotations[split] = {'path': name, 'sha256': sha256(data / name), 'samples': len(infos)}
    if counts != EXPECTED or len(all_tokens) != 34149:
        raise RuntimeError('Full trainval annotation coverage differs.')
    write_json(progress, {'stage': 'original_lidarseg_projected_depth', 'training_completed': False})
    depth_files = data / 'samples_point_label'
    existing_depth = {file.relative_to(data).as_posix() for file in depth_files.rglob('*.npy')}
    if len(expected_depth) != 34149 * 6 or existing_depth - expected_depth:
        raise RuntimeError('Projected depth filenames do not match every official camera annotation.')
    depth_generated = existing_depth != expected_depth
    if depth_generated:
        depth = load_module(source / 'tools/generate_point_label.py', 'official_full_depth')
        depth.generate(str(data), 'samples_point_label', version='v1.0-trainval')
    depth_count = 0
    for file in depth_files.rglob('*.npy'):
        depth = np.load(file, mmap_mode='r', allow_pickle=False)
        if depth.ndim != 2 or depth.shape[0] != 4 or not np.isfinite(depth).all():
            raise RuntimeError('Original projected depth format/content differs: ' + str(file))
        depth_count += 1
    if depth_count != 34149 * 6:
        raise RuntimeError('Incomplete original six-camera projected depth.')
    if {file.relative_to(data).as_posix() for file in depth_files.rglob('*.npy')} != expected_depth:
        raise RuntimeError('Original depth generation did not cover the exact annotated cameras.')
    write_json(progress, {'stage': 'hashing_all_model_local_inputs', 'training_completed': False})
    manifest = data / 'input_manifest.jsonl'
    temporary = manifest.with_suffix('.writing')
    count = 0
    with temporary.open('x') as stream:
        for base, dirs, files in os.walk(data):
            dirs.sort()
            for name in sorted(files):
                file = Path(base) / name
                if file in [manifest, temporary, progress, data / 'metadata.json']:
                    continue
                if not own_path(file, exists=True).is_relative_to(data):
                    raise RuntimeError('Manifest input escapes the independent dataset.')
                stream.write(json.dumps({'path': file.relative_to(data).as_posix(),
                    'bytes': file.stat().st_size, 'sha256': sha256(file)}) + '\n')
                count += 1
    temporary.replace(manifest)
    metadata = {'scope': SCOPE, 'counts': counts, 'scene_tokens': splits,
        'annotations': annotations, 'sensor_files': raw['sensor_files'], 'depth_files': depth_count,
        'official_can_fallback_samples': fallback,
        'can_policy': 'Pinned original converter fallback retained and counted; no project-invented CAN values',
        'manifest_files': count, 'manifest_sha256': sha256(manifest),
        'source_lock_sha256': sha256(ROOT / 'sources.lock.json'),
        'annotation_converter': 'pinned author vad_nuscenes_converter.py; trainval; max_sweeps=10',
        'converter_loading': converter_loading,
        'annotations_generated_this_invocation': annotations_generated,
        'depth_converter': 'pinned author generate_point_label.py; full lidarseg; all six cameras',
        'depth_generated_this_invocation': depth_generated,
        'full_training_completed': False, 'full_accuracy_validated': False}
    write_json(data / 'metadata.json', metadata)
    write_json(progress, {'stage': 'complete_inputs_prepared', 'training_completed': False})
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
