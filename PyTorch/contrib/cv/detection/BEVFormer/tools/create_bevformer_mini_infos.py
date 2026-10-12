#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Generate BEVFormer nuScenes mini infos without importing training operators.

The pinned upstream converter eagerly imports mmdet3d operators, even when it
only needs to write metadata. Execute its unchanged metadata functions and read
the installed mmdet3d category mapping as data. The 2D export path is omitted:
BEVFormer training consumes the temporal info PKLs generated here.
"""
from __future__ import annotations
import argparse
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
FUNCTIONS = {'create_nuscenes_infos', 'get_available_scenes', '_get_can_bus_info', '_fill_trainval_infos', 'obtain_sensor2top'}

def category_mapping(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'NuScenesDataset':
            for item in node.body:
                if isinstance(item, ast.Assign) and any((isinstance(target, ast.Name) and target.id == 'NameMapping' for target in item.targets)):
                    value = ast.literal_eval(item.value)
                    if not isinstance(value, dict) or len(value) < 10:
                        raise ValueError('Unexpected mmdet3d NuScenesDataset.NameMapping')
                    return value
    raise ValueError(f'NuScenesDataset.NameMapping missing in {path}')

def upstream_metadata_functions(path: Path, mapping: dict[str, str]):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected} != FUNCTIONS:
        raise ValueError('Pinned BEVFormer converter metadata functions changed')
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {'mmcv': mmcv, 'np': np, 'os': os, 'osp': os.path, 'NuScenes': NuScenes, 'Quaternion': Quaternion, 'NuScenesDataset': SimpleNamespace(NameMapping=mapping)}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['create_nuscenes_infos']

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Pinned, patched BEVFormer source checkout')
    parser.add_argument('--mmdet3d-dataset-source', type=Path, required=True, help='Installed mmdet3d/datasets/nuscenes_dataset.py')
    parser.add_argument('--canbus-root', type=Path, required=True, help='Directory containing can_bus, if available')
    parser.add_argument('--info-prefix', default='nuscenes')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    canbus_root = args.canbus_root.resolve(strict=True)
    converter = source / 'tools/data_converter/nuscenes_converter.py'
    data_root = source / 'data/nuscenes'
    for path in (converter, args.mmdet3d_dataset_source, data_root / 'v1.0-mini/scene.json', canbus_root):
        if not path.exists():
            parser.error(f'Missing input: {path}')
    create = upstream_metadata_functions(converter, category_mapping(args.mmdet3d_dataset_source))
    os.chdir(source)
    create(str(data_root), str(data_root), str(canbus_root), args.info_prefix, version='v1.0-mini')
    for split in ('train', 'val'):
        result = data_root / f'{args.info_prefix}_infos_temporal_{split}.pkl'
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError(f'Missing converted {split} infos: {result}')
        info = mmcv.load(str(result))
        if info.get('metadata', {}).get('version') != 'v1.0-mini':
            raise RuntimeError(f'Unexpected dataset version in {result}')
        print(f"{split}: {len(info['infos'])} samples, {result.stat().st_size} bytes")
if __name__ == '__main__':
    main()
