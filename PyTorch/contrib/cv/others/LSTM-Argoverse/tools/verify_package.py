"""Validate source locks and, when present, this entire offline package."""
import argparse
import json
from common import ROOT, sha256, own_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-only', action='store_true')
    args = parser.parse_args()
    count = 0
    source = json.loads((ROOT / 'sources.lock.json').read_text())
    for entry in source['files']:
        path = own_path(ROOT / entry['path'], exists=True)
        if sha256(path) != entry['sha256']:
            raise RuntimeError('固定源码 SHA256 不一致：' + entry['path'])
        count += 1
    if args.assets_only:
        archive = json.loads((ROOT / 'images.lock.json').read_text())['images']['runtime']['archive']
        path = own_path(ROOT / 'images' / archive['filename'], exists=True)
        if not path.is_file() or path.stat().st_size != archive['bytes'] or sha256(path) != archive['sha256']:
            raise RuntimeError('镜像归档缺失或校验失败。')
        count += 1
    elif (ROOT / 'package_manifest.json').is_file():
        manifest = json.loads((ROOT / 'package_manifest.json').read_text())
        for entry in manifest['files']:
            path = own_path(ROOT / entry['path'], exists=True)
            if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                raise RuntimeError('交付文件校验失败：' + entry['path'])
            count += 1
    print(json.dumps({'result': 'pass', 'checked_files': count}, indent=2))


if __name__ == '__main__':
    main()
