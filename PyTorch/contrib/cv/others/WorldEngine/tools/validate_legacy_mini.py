#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Check prepared functional data before the original training entry loads it."""
import json
import pickle
from pathlib import Path
from world_mini_data import ASSETS, require_hash

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
    for (_, _, path, _, checksum) in ASSETS:
        require_hash(Path(data_dir) / Path(path).name, checksum)
    annotations = [prepared / 'worldengine_navtest_smoke.pkl']
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
    return manifest
