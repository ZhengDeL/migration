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
    for item in manifest['labels']:
        path = prepared / item['path']
        if not path.resolve().is_relative_to(prepared):
            raise RuntimeError('Label path escapes prepared directory')
        require_hash(path, item['sha256'])
    annotations = [prepared / 'apollo_mini_train.pkl', prepared / 'apollo_mini_val.pkl']
    if manifest.get('heldout_samples', 0) and manifest.get('train_val_disjoint') is not True:
        raise RuntimeError('Apollo heldout split has not been checked for overlap')
    tokens = []
    for annotation in annotations:
        with annotation.open('rb') as stream:
            infos = pickle.load(stream)['infos']
        split_tokens = {info['token'] for info in infos}
        if len(split_tokens) != len(infos):
            raise RuntimeError('Duplicate sample tokens: ' + annotation.name)
        tokens.append(split_tokens)
        for info in infos:
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
    return manifest
