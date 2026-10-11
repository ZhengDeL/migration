#!/usr/bin/env python3
"""Verify and safely extract the full OCC release into a new directory."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import time
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
STAMP = '.migration_occ_manifest.json'
BLOCK = 4 * 1024 * 1024


class HashedReader:
    def __init__(self, stream, size):
        self.stream = stream
        self.size = size
        self.digest = hashlib.sha256()
        self.count = 0
        self.started = self.last_notice = time.monotonic()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        now = time.monotonic()
        if now - self.last_notice >= 20:
            print(f'OCC extraction: {self.count / 1e9:.2f}/{self.size / 1e9:.2f} GB compressed '
                  f'({100 * self.count / self.size:.1f}%)', flush=True)
            self.last_notice = now
        return data


def extract(archive_path, destination):
    spec = json.loads((ROOT / 'sources.lock.json').read_text(encoding='utf-8'))['apollo_visionnet']
    archive_path = Path(archive_path).resolve(strict=True)
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent or destination.is_symlink():
        raise RuntimeError('OCC destination must not traverse symbolic links')
    if destination.exists():
        stamp = destination / STAMP
        if stamp.is_file():
            previous = json.loads(stamp.read_text(encoding='utf-8'))
            if previous.get('archive_sha256_observed') == spec['occ_archive_sha256_observed'] and previous.get('gzip_crc_checked'):
                print('Verified OCC directory already exists: ' + str(destination), flush=True)
                return previous
        raise RuntimeError('Preserving an existing OCC directory without a matching verification stamp')
    before = archive_path.stat()
    if before.st_size != spec['occ_archive_bytes']:
        raise RuntimeError('OCC archive size differs from the verified full release')
    if shutil.disk_usage(destination.parent).free < spec['occ_uncompressed_file_bytes'] + 1024 ** 3:
        raise RuntimeError('OCC extraction needs at least 80 GB free disk space')
    staging = destination.parent / ('.' + destination.name + '.extracting-' + uuid.uuid4().hex[:12])
    staging.mkdir()
    print(f'Verifying and extracting full OCC into {staging.name}', flush=True)
    counts = Counter()
    seen = set()
    payload_bytes = 0
    try:
        with archive_path.open('rb', buffering=8 * 1024 * 1024) as raw:
            reader = HashedReader(raw, before.st_size)
            with gzip.GzipFile(fileobj=reader, mode='rb') as uncompressed:
                with tarfile.open(fileobj=uncompressed, mode='r|', bufsize=BLOCK) as archive:
                    for member in archive:
                        name = PurePosixPath(member.name)
                        if (name.is_absolute() or '..' in name.parts or '\\' in member.name
                                or not name.parts or name.parts[0] != 'occ_gt_release_v1_0'
                                or not (member.isfile() or member.isdir()) or name in seen):
                            raise RuntimeError('Unsafe or duplicate OCC archive member: ' + member.name)
                        seen.add(name)
                        counts['entries'] += 1
                        if counts['entries'] > spec['occ_archive_members']:
                            raise RuntimeError('Too many OCC archive members')
                        path = staging.joinpath(*name.parts[1:])
                        if not path.resolve().is_relative_to(staging):
                            raise RuntimeError('OCC archive member escapes destination')
                        if member.isdir():
                            path.mkdir(parents=True, exist_ok=True)
                            counts['directories'] += 1
                            continue
                        payload_bytes += member.size
                        if payload_bytes > spec['occ_uncompressed_file_bytes']:
                            raise RuntimeError('OCC payload exceeds the verified full release size')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, path.open('xb') as target:
                            remaining = member.size
                            while remaining:
                                block = source.read(min(BLOCK, remaining))
                                if not block:
                                    raise EOFError('Incomplete OCC archive member')
                                target.write(block)
                                remaining -= len(block)
                        counts['files'] += 1
                        counts['files_' + name.parts[1]] += 1
                # The tar reader stops at tar EOF; consuming gzip EOF verifies CRC/ISIZE.
                while uncompressed.read(BLOCK):
                    pass
            while reader.read(BLOCK):
                pass
            observed = reader.digest.hexdigest()
        after = archive_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('OCC archive changed during extraction')
        if observed != spec['occ_archive_sha256_observed'] or reader.count != before.st_size:
            raise RuntimeError('OCC archive SHA256 differs from the verified release; staging retained')
        if counts['entries'] != spec['occ_archive_members'] or payload_bytes != spec['occ_uncompressed_file_bytes']:
            raise RuntimeError('OCC inventory differs from the verified release')
        report = {'archive_name': archive_path.name, 'archive_bytes': before.st_size,
                  'archive_sha256_observed': observed, 'publisher_sha256': None,
                  'gzip_crc_checked': True, 'tar_members_checked': True,
                  'counts': dict(counts), 'uncompressed_file_bytes': payload_bytes,
                  'npy_arrays_checked': False, 'elapsed_seconds': round(time.monotonic() - reader.started, 2)}
        (staging / STAMP).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if destination.exists():
            raise RuntimeError('OCC destination appeared during extraction; staging retained')
        staging.rename(destination)
        print(json.dumps(report, indent=2), flush=True)
        return report
    except Exception:
        print('Incomplete extraction retained for inspection: ' + str(staging), flush=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    extract(args.archive, args.destination)
