"""Load only required definitions from exact, model-local upstream files."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from common import ROOT


def definitions(filename, names, namespace):
    source = ROOT / 'vendor' / filename
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    expected = next(item['sha256'] for item in lock['files']
                    if item['path'] == source.relative_to(ROOT).as_posix())
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Pinned upstream source SHA256 changed: ' + filename)
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('Required official definitions missing: ' + filename)
    namespace = dict(namespace, Any=Any, Dict=Dict, List=List, Tuple=Tuple,
                     math=math, copy=copy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def neural_classes():
    import torch
    result = definitions('lstm_train_test.py', ['EncoderRNN', 'DecoderRNN'],
                         {'torch':torch, 'nn':torch.nn, 'F':torch.nn.functional})
    return result['EncoderRNN'], result['DecoderRNN']


def coordinate_functions():
    import numpy as np
    import pandas as pd
    from shapely.geometry import LineString
    from shapely.affinity import affine_transform, rotate
    # These positions are from the exact official baseline_config.py snapshot.
    tree = ast.parse((ROOT / 'vendor/baseline_config.py').read_text())
    formats = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'FEATURE_FORMAT'
                           for target in node.targets))
    return definitions('baseline_utils.py', ['get_normalized_traj', 'get_relative_distance',
                                            'normalized_to_map_coordinates', 'get_abs_traj'],
                       {'np':np, 'pd':pd, 'LineString':LineString,
                        'affine_transform':affine_transform, 'rotate':rotate,
                        'FEATURE_FORMAT':formats})
