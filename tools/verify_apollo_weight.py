"""Verify the DLA checkpoint included in the pinned official Apollo Git tree."""
import subprocess
from pathlib import Path
from prepare import LOCK, require_checkout
from world_mini_data import require_hash


def verify(source):
    source = Path(source).resolve(strict=True)
    require_checkout('apollo_visionnet', source)
    spec = LOCK['apollo_visionnet']
    path = source / spec['pretrain_path']
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
        raise RuntimeError('Official Apollo DLA checkpoint must be a regular file in the source tree')
    if path.stat().st_size != spec['pretrain_bytes']:
        raise RuntimeError('Official Apollo DLA checkpoint size differs from the pinned Git tree')
    require_hash(path, spec['pretrain_sha256_observed'])
    blob = subprocess.check_output(['git', '-C', str(source), 'hash-object', str(path)], text=True).strip()
    if blob != spec['pretrain_git_blob_sha1']:
        raise RuntimeError('Official Apollo DLA checkpoint Git blob differs')
    return path
