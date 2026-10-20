"""Audit the original full Apollo annotations independently from raw media readiness."""
import json
from pathlib import Path
import pickle
import numpy as np
from world_mini_data import digest

NAMES = ('nuscenes_infos_temporal_train_occ_gt.pkl', 'nuscenes_infos_temporal_val_occ_gt.pkl')
LABEL_FIELDS = ('occ_gt_path', 'flow_gt_path', 'occ_invalid_path')


def label_path(value, occ):
    parts = Path(value).parts
    if 'occ_gt_release_v1_0' not in parts:
        raise RuntimeError('Original OCC path has no known dataset directory: ' + value)
    path = occ / Path(*parts[parts.index('occ_gt_release_v1_0') + 1:])
    if not path.is_file() or not path.resolve().is_relative_to(occ.resolve()):
        raise RuntimeError('Original OCC label/mask path missing or escapes its model-local root: ' + str(path))
    return path


def audit(paths, raw, occ, check_media):
    from nuscenes.utils import splits
    scenes = json.loads((raw / 'v1.0-trainval/scene.json').read_text())
    samples = json.loads((raw / 'v1.0-trainval/sample.json').read_text())
    if len(scenes) != 850 or len(samples) != 34149:
        raise RuntimeError('Expected complete official trainval metadata: 850 scenes / 34149 samples')
    scene_names = {x['token']: x['name'] for x in scenes}
    scene_for_sample = {x['token']: x['scene_token'] for x in samples}
    if len(scene_names) != 850 or len(scene_for_sample) != 34149:
        raise RuntimeError('Duplicate official full metadata tokens')
    records, token_sets = [], []
    for path, split, count in zip(paths, ('train', 'val'), (28130, 6019)):
        with path.open('rb') as stream:
            document = pickle.load(stream)
        infos = document['infos']
        tokens = {x['token'] for x in infos}
        allowed = set(getattr(splits, split))
        expected_tokens = {token for token, scene in scene_for_sample.items() if scene_names[scene] in allowed}
        if document.get('metadata', {}).get('version') != 'v1.0-trainval':
            raise RuntimeError('Full annotations must retain original trainval metadata')
        if len(infos) != count or len(tokens) != count or tokens != expected_tokens:
            raise RuntimeError('Original full split sample/token membership mismatch: ' + split)
        label_counts = {name: 0 for name in LABEL_FIELDS}
        media_references, nonzero_can = 0, 0
        for info in infos:
            if info['scene_token'] != scene_for_sample[info['token']]:
                raise RuntimeError('Annotation sample/scene relation differs from official metadata')
            if len(info['cams']) != 6:
                raise RuntimeError('Every full sample requires the original six cameras')
            can = np.asarray(info['can_bus'])
            if can.shape != (18,) or not np.isfinite(can).all():
                raise RuntimeError('Invalid original CAN vector')
            nonzero_can += int(bool(can.any()))
            for field in LABEL_FIELDS:
                if field not in info:
                    raise RuntimeError('Missing original label field: ' + field)
                actual = label_path(info[field], occ)
                relative = actual.relative_to(occ)
                if relative.parts[:2] != (split, scene_names[info['scene_token']]):
                    raise RuntimeError('OCC path belongs to another split/scene')
                label_counts[field] += 1
            if check_media:
                from prepare_data import rebase
                rebase(info['lidar_path'], raw)
                media_references += 1
                for camera in info['cams'].values():
                    rebase(camera['data_path'], raw)
                    media_references += 1
                for sweep in info['sweeps']:
                    rebase(sweep['data_path'], raw)
                    media_references += 1
        records.append(dict(split=split, frames=count, original_sha256=digest(path),
            label_references_checked=label_counts, nonzero_can_frames=nonzero_can,
            original_zero_can_frames=count - nonzero_can, media_references_checked=media_references))
        token_sets.append(tokens)
    if token_sets[0] & token_sets[1] or token_sets[0] | token_sets[1] != set(scene_for_sample):
        raise RuntimeError('Incomplete or overlapping original full trainval token coverage')
    return dict(result='pass_complete_original_full_annotation_membership_and_label_paths',
        splits=records, total_samples=34149, train_val_disjoint=True,
        labels_and_masks_checked=102447, raw_media_coverage_checked=check_media,
        referenced_media_present=check_media, training_preparation_completed=False,
        formal_accuracy_accepted=False)
