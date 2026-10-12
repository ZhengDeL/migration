"""Utilities contained in this model's own directory."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.partial')
    part.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    part.replace(path)


def own_path(path, exists=False):
    resolved = Path(path).resolve(strict=exists)
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError('路径必须在本模型目录内：' + str(resolved))
    return resolved


def tensor_tree_cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: tensor_tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tensor_tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree_cpu(v) for v in value)
    return value


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def parameter_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode('utf-8'))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
