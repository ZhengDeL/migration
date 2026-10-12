#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Verify pinned upstream source and apply the reviewed Ascend patch."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'sources.lock.json').read_text(encoding='utf-8'))
MODELS = tuple((name for name in LOCK if name != 'drivingsdk'))
SPARSE_CLIP_ORIGINAL = b'return runner.optimizer.clip_grad_norm_fused_(**self.grad_clip)'
SPARSE_CLIP_REPLACEMENT = b'return torch.nn.utils.clip_grad_norm_(params, **self.grad_clip)'
SPARSE_OPT_ORIGINAL = b"optimizer_cfg['type'] = 'NpuFused' + optimizer_cfg['type']"
SPARSE_OPT_REPLACEMENT = b'# Preserve the configured optimizer type (AdamW).'

def sparse_helper_bytes(supplied: Path) -> bytes:
    content = supplied.read_bytes()
    replacements = ((SPARSE_CLIP_ORIGINAL, SPARSE_CLIP_REPLACEMENT), (SPARSE_OPT_ORIGINAL, SPARSE_OPT_REPLACEMENT))
    for (original, replacement) in replacements:
        if content.count(original) != 1:
            raise RuntimeError(f'{supplied}: expected one pinned occurrence of {original!r}')
        content = content.replace(original, replacement)
    return content

def run(*args: str, cwd: Path, check: bool=True, env: dict[str, str] | None=None, input: bytes | None=None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, env=env, input=input, capture_output=True, check=check)

def require_checkout(name: str, source: Path) -> None:
    if not (source / '.git').exists():
        raise RuntimeError(f'{source}: expected a Git checkout')
    expected = LOCK[name]
    head = run('git', 'rev-parse', 'HEAD', cwd=source).stdout.decode().strip()
    if head != expected['revision']:
        raise RuntimeError(f"{name}: HEAD {head}, expected {expected['revision']}")
    origin = run('git', 'remote', 'get-url', 'origin', cwd=source).stdout.decode().strip()
    if origin.rstrip('/').removesuffix('.git').lower() != expected['url'].removesuffix('.git').lower():
        raise RuntimeError(f'{name}: unexpected origin {origin}')

def patch_for(name: str, sdk: Path | None) -> Path | None:
    spec = LOCK[name].get('patch')
    if not spec:
        return None
    if spec.startswith('drivingsdk:'):
        if sdk is None:
            raise RuntimeError('--sdk is required for DrivingSDK model patches')
        require_checkout('drivingsdk', sdk)
        path = sdk / spec.partition(':')[2]
    else:
        path = ROOT / spec
    if not path.is_file():
        raise RuntimeError(f'missing patch: {path}')
    digest = hashlib.sha256(patch_bytes(path)).hexdigest()
    if digest != LOCK[name]['patch_sha256']:
        raise RuntimeError(f'{name}: patch SHA256 {digest} does not match sources.lock.json')
    return path

def compat_patch_for(name: str) -> Path | None:
    spec = LOCK[name].get('compat_patch')
    if not spec:
        return None
    path = ROOT / spec
    if not path.is_file():
        raise RuntimeError(f'missing compatibility patch: {path}')
    digest = hashlib.sha256(patch_bytes(path)).hexdigest()
    if digest != LOCK[name]['compat_patch_sha256']:
        raise RuntimeError(f'{name}: compatibility patch SHA256 {digest} does not match sources.lock.json')
    return path

def patch_bytes(patch: Path) -> bytes:
    return patch.read_bytes().replace(b'\r\n', b'\n')

def patch_state(source: Path, patch: Path) -> str:
    data = patch_bytes(patch)
    if run('git', 'apply', '--reverse', '--check', '--ignore-space-change', '-', cwd=source, input=data, check=False).returncode == 0:
        return 'applied'
    result = run('git', 'apply', '--check', '--ignore-space-change', '-', cwd=source, input=data, check=False)
    if result.returncode == 0:
        return 'ready'
    raise RuntimeError(f"patch does not apply cleanly to {source}: {result.stderr.decode(errors='replace').strip()}")

def verify_applied_patch(source: Path, patch: Path, extra_patches: tuple[Path, ...]=()) -> None:
    """Compare every patched blob to patches applied to a temporary HEAD index."""
    if run('git', 'diff', '--cached', '--quiet', 'HEAD', cwd=source, check=False).returncode:
        raise RuntimeError(f'{source}: staged source changes are not allowed')
    with tempfile.TemporaryDirectory(prefix='ascend-patch-index-') as temp:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(temp) / 'index')}
        run('git', 'read-tree', 'HEAD', cwd=source, env=env)
        for item in (patch, *extra_patches):
            applied = run('git', 'apply', '--cached', '--ignore-space-change', '-', cwd=source, env=env, input=patch_bytes(item), check=False)
            if applied.returncode:
                raise RuntimeError(f"{source}: cannot reconstruct reviewed patch {item} from HEAD: {applied.stderr.decode(errors='replace').strip()}")
        expected_paths = set(run('git', 'diff', '--cached', '--name-only', 'HEAD', cwd=source, env=env).stdout.decode().splitlines())
        actual_paths = set(run('git', 'diff', '--name-only', 'HEAD', cwd=source).stdout.decode().splitlines())
        for relative in expected_paths - actual_paths:
            in_head = run('git', 'cat-file', '-e', f'HEAD:{relative}', cwd=source, check=False).returncode == 0
            if not in_head and (source / relative).is_file():
                actual_paths.add(relative)
        if actual_paths != expected_paths:
            raise RuntimeError(f'{source}: tracked changes differ from reviewed patch; extra={sorted(actual_paths - expected_paths)}, missing={sorted(expected_paths - actual_paths)}')
        for relative in sorted(expected_paths):
            expected = run('git', 'show', f':{relative}', cwd=source, env=env).stdout
            path = source / relative
            if not path.is_file():
                raise RuntimeError(f'{path}: patched file missing')
            actual = path.read_bytes()
            normalize = lambda content: content.replace(b'\r\n', b'\n').rstrip(b'\r')
            if normalize(actual) != normalize(expected):
                raise RuntimeError(f'{path}: content differs from reviewed patch')

def stage_sparse4d_files(source: Path, sdk: Path) -> None:
    official = sdk / 'model_examples' / 'Sparse4D'
    target = source / 'tools' / 'patch.py'
    supplied = official / 'patch.py'
    expected = sparse_helper_bytes(supplied)
    if target.exists() and target.read_bytes() not in (supplied.read_bytes(), supplied.read_bytes().replace(SPARSE_CLIP_ORIGINAL, SPARSE_CLIP_REPLACEMENT), expected):
        raise RuntimeError(f'refusing to replace nonmatching {target}')
    target.write_bytes(expected)
    test_target = source / 'test'
    test_target.mkdir(exist_ok=True)
    for script in (official / 'test').glob('*.sh'):
        dest = test_target / script.name
        if dest.exists() and dest.read_bytes() != script.read_bytes():
            raise RuntimeError(f'refusing to replace nonmatching {dest}')
        shutil.copy2(script, dest)

def verify_sparse4d_files(source: Path, sdk: Path) -> None:
    official = sdk / 'model_examples' / 'Sparse4D'
    files = [(official / 'patch.py', source / 'tools' / 'patch.py')]
    files.extend(((script, source / 'test' / script.name) for script in (official / 'test').glob('*.sh')))
    for (supplied, target) in files:
        expected = sparse_helper_bytes(supplied) if supplied.name == 'patch.py' else supplied.read_bytes()
        if not supplied.is_file() or not target.is_file() or expected != target.read_bytes():
            raise RuntimeError(f'{target}: missing or differs from pinned DrivingSDK copy')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--source', required=True, type=Path, help='existing pinned upstream checkout')
    parser.add_argument('--sdk', type=Path, help='existing DrivingSDK checkout at locked revision')
    parser.add_argument('--apply', action='store_true', help='apply patch after verification')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    sdk = args.sdk.resolve(strict=True) if args.sdk else None
    require_checkout(args.model, source)
    patch = patch_for(args.model, sdk)
    compat = compat_patch_for(args.model)
    if patch is None:
        print(json.dumps({'model': args.model, 'revision': LOCK[args.model]['revision'], 'state': 'upstream_ready'}))
        return 0
    state = patch_state(source, patch)
    if args.apply and state == 'ready':
        dirty = run('git', 'status', '--porcelain', cwd=source).stdout.strip()
        if dirty:
            raise RuntimeError(f'{source} has unrelated changes; use a clean checkout')
        run('git', 'apply', '--ignore-space-change', '-', cwd=source, input=patch_bytes(patch))
        state = patch_state(source, patch)
    compat_state = None
    if compat is not None and state == 'applied':
        compat_state = patch_state(source, compat)
        if args.apply and compat_state == 'ready':
            run('git', 'apply', '--ignore-space-change', '-', cwd=source, input=patch_bytes(compat))
            compat_state = patch_state(source, compat)
    if state == 'applied':
        if compat is None or compat_state == 'applied':
            verify_applied_patch(source, patch, (compat,) if compat else ())
        elif args.apply:
            raise RuntimeError(f'{args.model}: compatibility patch was not applied')
    print(json.dumps({'model': args.model, 'revision': LOCK[args.model]['revision'], 'patch_sha256': hashlib.sha256(patch_bytes(patch)).hexdigest(), 'state': state, 'compat_state': compat_state}))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(f'prepare failed: {exc}')
