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
    for name, checksum in manifest['prepared_files'].items():
        path = prepared / name
        if not path.resolve().is_relative_to(prepared) or not path.is_file():
            raise RuntimeError('Invalid prepared data file: ' + name)
        require_hash(path, checksum)
    if model == 'worldengine':
        for _, _, path, _, checksum in ASSETS:
            require_hash(Path(data_dir) / Path(path).name, checksum)
        annotation = prepared / 'worldengine_navtest_smoke.pkl'
    else:
        if not manifest.get('label_payloads_unchanged'):
            raise RuntimeError('Apollo label verification is missing')
        for item in manifest['labels']:
            path = prepared / item['path']
            if not path.resolve().is_relative_to(prepared):
                raise RuntimeError('Label path escapes prepared directory')
            require_hash(path, item['sha256'])
        annotation = prepared / 'apollo_mini_train.pkl'
    # Only load files just matched to the locally prepared manifest above.
    with annotation.open('rb') as stream:
        infos = pickle.load(stream)['infos']
    for info in infos:
        for camera in info['cams'].values():
            if not Path(camera['data_path']).is_file():
                raise RuntimeError('Missing real image: ' + camera['data_path'])
    return manifest
