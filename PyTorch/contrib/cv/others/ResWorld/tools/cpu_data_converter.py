"""Load the author's metadata converter without importing NPU model operators.

The trainval converter uses only NuScenesDataset.NameMapping. Its optional 2D
export also imports points_cam2img; that separate export is deliberately refused.
The mapping is extracted from the actual pinned image's dataset source by AST.
No conversion algorithm or upstream/vendor file is rewritten.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from common import sha256


def load_author_converter(path):
    spec = importlib.util.find_spec('mmdet3d')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('The pinned image must contain the original MMDet3D dataset source.')
    mapping_source = Path(next(iter(spec.submodule_search_locations))) / 'datasets/nuscenes_dataset.py'
    tree = ast.parse(mapping_source.read_text(), filename=str(mapping_source))
    dataset = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'NuScenesDataset')
    assignments = [node for node in dataset.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'NameMapping'
                           for target in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError('Review the changed original NuScenes category mapping.')
    mapping = ast.literal_eval(assignments[0].value)
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                              for k, v in mapping.items()):
        raise RuntimeError('Expected a literal original category-name mapping.')
    path = Path(path)
    code = ast.parse(path.read_text(), filename=str(path))
    allowed = {'mmdet3d.datasets': 'NuScenesDataset',
               'mmdet3d.core.bbox.box_np_ops': 'points_cam2img'}
    original_functions = ast.dump(ast.Module(body=[node for node in code.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]))
    removed, body, cli_nodes = {}, [], 0
    for node in code.body:
        if isinstance(node, ast.ImportFrom) and node.module in allowed:
            if len(node.names) != 1 or node.names[0].name != allowed[node.module] or node.names[0].asname:
                raise RuntimeError('Review changed converter imports before CPU preparation.')
            removed[node.module] = node.names[0].name
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['parser', 'args']
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and (node.value.func.value.id, node.value.func.attr) in
              [('argparse', 'ArgumentParser'), ('parser', 'parse_args')]):
            cli_nodes += 1
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name)
              and (node.value.func.value.id, node.value.func.attr) == ('parser', 'add_argument')):
            cli_nodes += 1
        else:
            body.append(node)
    if removed != allowed or cli_nodes != 10:
        raise RuntimeError('Expected exactly the two known model-only converter imports.')
    code.body = body
    actual_functions = ast.dump(ast.Module(body=[node for node in code.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]))
    if actual_functions != original_functions:
        raise RuntimeError('The original conversion functions must remain unchanged.')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
        'resworld_cpu_original_metadata_converter', path))
    module.NuScenesDataset = SimpleNamespace(NameMapping=mapping)

    def unsupported_2d_export(*args, **kwargs):
        raise RuntimeError('CPU metadata preparation does not implement the separate optional 2D export.')

    module.points_cam2img = unsupported_2d_export
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return module, {'mode': 'original_converter_with_metadata_only_imports',
        'converter_sha256': sha256(path), 'mapping_source': str(mapping_source),
        'mapping_source_sha256': sha256(mapping_source), 'mapping_entries': len(mapping),
        'isolated_imports': removed, 'optional_2d_export': 'refused',
        'import_time_cli_parser_nodes_omitted': cli_nodes,
        'original_function_ast_unchanged': True,
        'conversion_algorithm_changed': False}
