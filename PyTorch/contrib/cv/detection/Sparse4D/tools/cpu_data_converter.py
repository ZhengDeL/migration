"""Load pinned author data/anchor functions without importing NPU model operators."""
import ast
import hashlib
import importlib.util
from pathlib import Path


def load_author_converter(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location('sparse4d_original_cpu_converter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(converter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        conversion_functions_unchanged=True)


def load_author_anchor_generator(path, box_path):
    path, box_path = Path(path), Path(box_path)
    box_tree = ast.parse(box_path.read_text(), filename=str(box_path))
    # The pinned file contains only the original numeric box-index assignments.
    if any(not isinstance(node, ast.Assign) for node in box_tree.body):
        raise RuntimeError('Review changed original box constants before CPU anchor generation')
    constants = {'__builtins__': {'list': list, 'range': range}}
    exec(compile(box_tree, str(box_path), 'exec'), constants)
    expected = dict(zip(('X', 'Y', 'Z', 'W', 'L', 'H', 'SIN_YAW', 'COS_YAW', 'VX', 'VY', 'VZ'), range(11)))
    if any(constants.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Original box constant indices changed')
    tree = ast.parse(path.read_text(), filename=str(path))
    original_functions = ast.dump(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]))
    removed, retained = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'projects.mmdet3d_plugin.core.box3d':
            if [(name.name, name.asname) for name in node.names] != [('*', None)]:
                raise RuntimeError('Review changed original box imports')
            removed.append(node.module)
        else:
            retained.append(node)
    if removed != ['projects.mmdet3d_plugin.core.box3d']:
        raise RuntimeError('Expected exactly the original model-only box-index import')
    tree.body = retained
    if ast.dump(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])) != original_functions:
        raise RuntimeError('Original anchor function must remain unchanged')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('sparse4d_original_cpu_anchor', path))
    module.__dict__.update({key: value for key, value in constants.items() if key != '__builtins__'})
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module, dict(anchor_generator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        box_constants_sha256=hashlib.sha256(box_path.read_bytes()).hexdigest(),
        removed_model_only_import=removed[0], anchor_functions_unchanged=True)
