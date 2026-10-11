#!/usr/bin/env python3
"""Verify a split offline delivery and restore it to an empty Linux workspace."""
import argparse
import hashlib
import json
import os
import platform
import posixpath
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path, PurePosixPath

CHUNK = 8 * 1024 * 1024


def relative_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise RuntimeError('Unsafe package path: ' + name)
    return path


def below(root, name):
    path = root / relative_name(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Package path escapes its directory: ' + name)
    return path


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(bundle):
    path = bundle / 'manifest.json'
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('schema_version') != 1 or not manifest.get('files') or not manifest.get('bundles'):
        raise RuntimeError('Unknown or incomplete delivery manifest')
    names = [item['path'] for item in manifest['files']]
    if len(names) != len(set(names)):
        raise RuntimeError('Duplicate files in delivery manifest')
    for item in manifest['files']:
        relative_name(item['path'])
        if item['bytes'] < 0 or len(item['sha256']) != 64:
            raise RuntimeError('Invalid file checksum metadata')
    files = {item['path']: item for item in manifest['files']}
    for archive in manifest['bundles']:
        if not archive['parts'] or len(set(archive['parts'])) != len(archive['parts']):
            raise RuntimeError('Invalid archive parts')
        if any(name not in files for name in archive['parts']):
            raise RuntimeError('An archive part is absent from the manifest')
        if sum(files[name]['bytes'] for name in archive['parts']) != archive['bytes']:
            raise RuntimeError('Archive part sizes do not match its total')
    return manifest, hashlib.sha256(raw).hexdigest()


def verify(bundle, manifest):
    files = {item['path']: item for item in manifest['files']}
    checked = set()
    for archive in manifest['bundles']:
        combined = hashlib.sha256()
        for name in archive['parts']:
            path = below(bundle, name)
            item = files[name]
            if not path.is_file() or path.is_symlink() or path.stat().st_size != item['bytes']:
                raise RuntimeError('Missing or incomplete part: ' + name)
            part_hash = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(CHUNK), b''):
                    part_hash.update(block)
                    combined.update(block)
            if part_hash.hexdigest() != item['sha256']:
                raise RuntimeError('SHA256 mismatch: ' + name)
            checked.add(name)
            print('Verified ' + name, flush=True)
        if combined.hexdigest() != archive['sha256']:
            raise RuntimeError('Combined archive SHA256 mismatch: ' + archive['id'])
    for name, item in files.items():
        if name in checked:
            continue
        path = below(bundle, name)
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item['bytes'] or checksum(path) != item['sha256']:
            raise RuntimeError('Delivery support file failed checksum: ' + name)
    print(f"Delivery verification passed: {len(files)} files, {len(manifest['bundles'])} archives", flush=True)


def assemble(bundle, archive, cache):
    filename = relative_name(archive['archive_name'])
    if len(filename.parts) != 1:
        raise RuntimeError('Archive cache name must be a basename')
    output = cache / filename
    if output.exists():
        if not output.is_file() or output.is_symlink() or output.stat().st_size != archive['bytes'] or checksum(output) != archive['sha256']:
            raise RuntimeError('Existing archive cache differs; preserve and inspect it: ' + str(output))
        return output
    temporary = output.with_name(output.name + '.assembling')
    if temporary.is_symlink():
        raise RuntimeError('Invalid assembly temporary path')
    digest = hashlib.sha256()
    with temporary.open('wb') as target:
        for name in archive['parts']:
            with below(bundle, name).open('rb') as source:
                for block in iter(lambda: source.read(CHUNK), b''):
                    target.write(block)
                    digest.update(block)
    if temporary.stat().st_size != archive['bytes'] or digest.hexdigest() != archive['sha256']:
        raise RuntimeError('Reassembled archive checksum failed: ' + archive['id'])
    temporary.rename(output)
    return output


def extract(archive, root):
    """Only create ordinary files, directories, and relative links within root."""
    seen, links = set(), []
    with tarfile.open(str(archive), mode='r|*') as stream:
        for member in stream:
            name = relative_name(member.name).as_posix()
            if name == '.':
                continue
            if name in seen:
                raise RuntimeError('Duplicate archive member: ' + name)
            seen.add(name)
            target = below(root, name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise RuntimeError('Missing archive payload: ' + name)
                with source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination, CHUNK)
                os.chmod(target, member.mode & 0o777)
            elif member.issym():
                if PurePosixPath(member.linkname).is_absolute() or '\\' in member.linkname:
                    raise RuntimeError('Absolute or invalid archive symlink: ' + name)
                linked = posixpath.normpath(posixpath.join(posixpath.dirname(name), member.linkname))
                relative_name(linked)
                links.append((target, member.linkname))
            else:
                raise RuntimeError('Unsupported archive member type: ' + name)
    for target, linked in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise RuntimeError('Archive symlink would overwrite a file: ' + str(target))
        target.symlink_to(linked)
        if not target.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('Archive symlink escapes restored workspace')


def image_check(images):
    for image in images:
        result = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', image['tag']],
                                text=True, capture_output=True)
        if result.returncode == 0 and result.stdout.strip() != image['image_id'] + '|arm64':
            raise RuntimeError('Existing Docker tag differs from the delivery: ' + image['tag'])


def rebase(root, previous):
    # These are generated metadata from the checksum-verified delivery, never arbitrary downloaded PKLs.
    import pickle
    names = [
        'data/nuscenes/nuscenes_infos_temporal_train.pkl',
        'data/nuscenes/nuscenes_infos_temporal_val.pkl',
        'migration/sources/Sparse4D/data/nuscenes_anno_pkls/nuscenes_infos_train.pkl',
        'migration/sources/Sparse4D/data/nuscenes_anno_pkls/nuscenes_infos_val.pkl',
    ]

    def replace(value):
        if isinstance(value, str) and (value == previous or value.startswith(previous + '/')):
            return str(root) + value[len(previous):]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        return value

    for name in names:
        path = below(root, name)
        if not path.is_file():
            raise RuntimeError('Required generated metadata is missing: ' + name)
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        temporary = path.with_name(path.name + '.rebasing')
        with temporary.open('wb') as stream:
            pickle.dump(replace(saved), stream, protocol=4)
        temporary.replace(path)
        print('Prepared metadata paths: ' + name, flush=True)


def prepare(root, manifest):
    image = next(item['tag'] for item in manifest['images'] if item['role'] == 'drivingsdk_headless')
    command = ['docker', 'run', '--rm', '--network', 'none', '--memory', '8g', '--cpus', '2',
               '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-v', str(root) + ':' + str(root) + ':rw',
               '--entrypoint', '/opt/conda/envs/bevformer/bin/python', image,
               str(root / 'migration/tools/restore_delivery.py'), 'rebase', '--destination', str(root),
               '--original-root', manifest['original_root']]
    subprocess.run(command, check=True)
    for model in ('worldengine', 'apollo_visionnet'):
        subprocess.run(['bash', str(root / 'migration/tools/quickstart.sh'), model, 'prepare'],
                       cwd=root / 'migration', check=True)
    # Recreate the same mini configuration with the restored paths for saved-checkpoint checks.
    for model, source, data in (
        ('apollo_visionnet', 'Apollo-Vision-Net', 'apollo'),
        ('worldengine', 'WorldEngine', 'worldengine_smoke'),
    ):
        selected = next(item['tag'] for item in manifest['images']
                        if item['role'] == ('worldengine' if model == 'worldengine' else 'drivingsdk_headless'))
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--memory', '8g', '--cpus', '2',
               '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-v', str(root) + ':' + str(root) + ':rw',
               '-w', str(root / 'migration'), '--entrypoint', '/opt/conda/envs/bevformer/bin/python', selected,
               'tools/create_legacy_mini_config.py', model, '--repo', str(root / 'migration'),
               '--source', str(root / 'migration/sources' / source), '--data-dir', str(root / 'data' / data),
               '--output-dir', str(root / 'migration/outputs' / model / 'mini')]
        if model == 'apollo_visionnet':
            cmd += ['--prepared-dir', str(root / 'data/apollo/mini_prepared'), '--nuscenes', str(root / 'data/nuscenes')]
        subprocess.run(cmd, check=True)


def restore(bundle, manifest, manifest_hash, destination):
    if os.name != 'posix' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Restore requires an ARM64 Linux host with Docker; verify works on Windows too')
    destination = destination.absolute()
    if destination.is_symlink():
        raise RuntimeError('Destination must not be a symlink')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    marker = destination / '.migration_delivery.json'
    resume = False
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('manifest_sha256') != manifest_hash:
            raise RuntimeError('Destination is not empty and is not owned by this delivery: ' + str(destination))
        resume = True
    if shutil.disk_usage(destination.parent).free < manifest['minimum_free_bytes'] and not resume:
        raise RuntimeError('Insufficient free space; see the delivery README')
    image_check(manifest['images'])
    cache = destination.parent / ('.' + destination.name + '.delivery_cache_' + manifest_hash[:12])
    cache.mkdir(exist_ok=True)
    if cache.is_symlink():
        raise RuntimeError('Invalid delivery cache directory')
    archives = {item['id']: assemble(bundle, item, cache) for item in manifest['bundles']}
    if not resume:
        staging = destination.parent / ('.' + destination.name + '.restoring-' + uuid.uuid4().hex[:12])
        staging.mkdir()
        for archive in manifest['bundles']:
            if archive['kind'] == 'workspace':
                print('Restoring ' + archive['id'], flush=True)
                extract(archives[archive['id']], staging)
            elif archive['kind'] == 'file':
                target = below(staging, archive['destination'])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise RuntimeError('Raw archive target already exists')
                shutil.copyfile(archives[archive['id']], target)
            elif archive['kind'] != 'images':
                raise RuntimeError('Unknown archive restore kind')
        # The public mini archive contains files rooted at samples/, sweeps/, maps/, v1.0-mini/.
        nuscenes = staging / 'data/nuscenes'
        print('Extracting checksum-verified nuScenes mini', flush=True)
        extract(nuscenes / 'v1.0-mini.tgz', nuscenes)
        (staging / '.migration_delivery.json').write_text(json.dumps({'manifest_sha256': manifest_hash,
                                                                   'package_id': manifest['package_id'],
                                                                   'status': 'data_published_prepare_pending'}, indent=2) + '\n')
        if destination.exists():
            if any(destination.iterdir()):
                raise RuntimeError('Destination changed during restore')
            destination.rmdir()  # Only the empty destination checked above.
        staging.rename(destination)
    for archive in manifest['bundles']:
        if archive['kind'] == 'images':
            print('Loading checksum-verified Docker images', flush=True)
            subprocess.run(['docker', 'load', '-i', str(archives[archive['id']])], check=True)
    image_check(manifest['images'])
    for image in manifest['images']:
        subprocess.run(['docker', 'image', 'inspect', image['tag']], check=True, stdout=subprocess.DEVNULL)
    prepare(destination, manifest)
    marker.write_text(json.dumps({'manifest_sha256': manifest_hash, 'package_id': manifest['package_id'],
                                  'status': 'restored'}, indent=2) + '\n')
    print('Offline restore completed: ' + str(destination / 'migration'), flush=True)
    print('Next: cd ' + str(destination / 'migration') + '\nbash tools/quickstart.sh bevformer plan', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'restore', 'rebase'))
    parser.add_argument('--bundle', type=Path, default=Path.cwd())
    parser.add_argument('--destination', type=Path, default=Path('/var/ai-model/lzd'))
    parser.add_argument('--original-root', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == 'rebase':
        if not args.original_root:
            parser.error('--original-root is required for the internal metadata preparation step')
        rebase(args.destination.resolve(strict=True), args.original_root)
        return
    bundle = args.bundle.resolve(strict=True)
    manifest, manifest_hash = load_manifest(bundle)
    verify(bundle, manifest)
    if args.action == 'restore':
        restore(bundle, manifest, manifest_hash, args.destination)


if __name__ == '__main__':
    main()
