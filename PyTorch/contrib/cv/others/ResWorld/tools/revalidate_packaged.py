"""Verify one complete offline package and revalidate its checkpoint in a fresh output directory."""
import argparse
import datetime
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, own_path, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0', help='one host logical NPU ID')
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ['aarch64', 'arm64']:
        raise RuntimeError('请在 ARM64 Linux Ascend 服务器运行本模型离线包。')
    if not args.devices.isdigit():
        raise RuntimeError('单卡复验需要一个宿主逻辑 NPU ID。')
    manifest_path = ROOT / 'package_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('请先完整解压本模型离线 ZIP；GitHub 源码目录不包含交付检查点。')
    manifest = json.loads(manifest_path.read_text())
    preferred = manifest.get('preferred_output_dir')
    if not preferred:
        raise RuntimeError('离线包版本缺少当前检查点入口，请使用本模型最新完整归档。')
    source = own_path(ROOT / preferred, exists=True)
    output = own_path(args.output_dir) if args.output_dir else ROOT / 'test/output' / ('packaged_recheck_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        raise RuntimeError('复验输出目录已存在，请选择新目录。')
    subprocess.run([sys.executable, str(ROOT / 'tools/verify_package.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/container.py'), 'setup',
                    '--nproc', '1', '--devices', args.devices], check=True)
    output.mkdir(parents=True, exist_ok=False)
    for name in ['checkpoint.pt', 'reload_reference.pt', 'train_result.json']:
        shutil.copy2(source / name, output / name)
    code = subprocess.call([sys.executable, str(ROOT / 'tools/container.py'), 'validate',
                            '--devices', args.devices, '--output_dir', str(output)])
    result = {'result': 'pass_packaged_checkpoint_revalidation' if code == 0 else 'fail',
              'exit_code': code, 'source_output_dir': preferred,
              'new_output_dir': output.relative_to(ROOT).as_posix(),
              'package_manifest_sha256': sha256(manifest_path),
              'source_checkpoint_sha256': sha256(source / 'checkpoint.pt'),
              'copied_checkpoint_sha256': sha256(output / 'checkpoint.pt')}
    write_json(output / 'packaged_recheck_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
