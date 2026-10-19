#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Check prepared functional data before the original training entry loads it."""
import json
import pickle
from pathlib import Path
from world_mini_data import require_hash

def validate(model, prepared, data_dir):
    prepared = Path(prepared).resolve(strict=True)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if manifest.get('model') != model or not manifest.get('prepared_files'):
        raise RuntimeError('Run the model data preparation command first')
    for (name, checksum) in manifest['prepared_files'].items():
        path = prepared / name
        if not path.resolve().is_relative_to(prepared) or not path.is_file():
            raise RuntimeError('Invalid prepared data file: ' + name)
        require_hash(path, checksum)
    if not manifest.get('label_payloads_unchanged'):
        raise RuntimeError('Apollo label verification is missing')
    for item in manifest['labels'] + manifest.get('invalid_masks', []):
        path = prepared / item['path']
        if not path.resolve().is_relative_to(prepared):
            raise RuntimeError('Label path escapes prepared directory')
        require_hash(path, item['sha256'])
    annotations = [prepared / 'apollo_mini_train.pkl', prepared / 'apollo_mini_val.pkl']
    if manifest.get('heldout_samples', 0) and manifest.get('train_val_disjoint') is not True:
        raise RuntimeError('Apollo heldout split has not been checked for overlap')
    tokens = []
    invalid_paths = set()
    for annotation in annotations:
        with annotation.open('rb') as stream:
            infos = pickle.load(stream)['infos']
        split_tokens = {info['token'] for info in infos}
        if len(split_tokens) != len(infos):
            raise RuntimeError('Duplicate sample tokens: ' + annotation.name)
        tokens.append(split_tokens)
        for info in infos:
            if 'occ_invalid_path' in info:
                invalid = Path(info['occ_invalid_path'])
                if (not manifest.get('invalid_masks_payloads_unchanged') or
                        not invalid.resolve().is_relative_to(prepared) or not invalid.is_file()):
                    raise RuntimeError('Run prepare_data to restore and verify original OCC visibility masks')
                invalid_paths.add(str(invalid.relative_to(prepared)))
            for camera in info['cams'].values():
                if not Path(camera['data_path']).is_file():
                    raise RuntimeError('Missing real image: ' + camera['data_path'])
    if len(tokens[0]) != manifest['train_samples']:
        raise RuntimeError('Apollo train sample count differs from manifest')
    if manifest.get('heldout_samples', 0):
        if len(tokens[1]) != manifest['heldout_samples'] or tokens[0] & tokens[1]:
            raise RuntimeError('Apollo validation count or train/val overlap is invalid')
        if len(tokens[0] | tokens[1]) != manifest['mini_samples_expected']:
            raise RuntimeError('Apollo mini split does not cover the expected samples')
    if invalid_paths != {item['path'] for item in manifest.get('invalid_masks', [])}:
        raise RuntimeError('Original OCC visibility mask references differ from verified manifest')
    return manifest
