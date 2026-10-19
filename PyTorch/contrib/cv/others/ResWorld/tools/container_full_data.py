"""Prepare this model's own complete data in the pinned image, without NPU devices."""
import argparse
import json
import subprocess
from pathlib import Path
from common import ROOT, PROFILE, own_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', type=Path, default=ROOT / 'data/nuscenes_full')
    p.add_argument('--check-mini-components', action='store_true')
    p.add_argument('--output_dir', type=Path)
    a = p.parse_args()
    data = own_path(a.data_path, exists=True)
    if a.check_mini_components and a.output_dir is None:
        p.error('--check-mini-components requires a new --output_dir.')
    if a.output_dir is not None and not a.check_mini_components:
        p.error('--output_dir is used only with --check-mini-components.')
    spec = json.loads((ROOT / 'images.lock.json').read_text())['images']['runtime']
    actual = subprocess.check_output(['docker', 'image', 'inspect', '--format',
        '{{.Id}}|{{.Architecture}}', spec['tag']], text=True).strip()
    if actual != spec['image_id'] + '|arm64':
        raise RuntimeError('The pinned full-data preparation image differs.')
    import fcntl
    with (ROOT / '.model_operation.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--memory', '64g', '--cpus', '8',
            '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-e', 'OMP_NUM_THREADS=2',
            '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'PYTHONUNBUFFERED=1',
            '-v', str(ROOT) + ':' + str(ROOT) + ':rw', '-w', str(ROOT)]
        for path in ['/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi']:
            if Path(path).exists():
                cmd += ['-v', path + ':' + path + ':ro']
        entry = 'tools/validate_cpu_data_converter.py' if a.check_mini_components else 'tools/prepare_full_data.py'
        cmd += ['--entrypoint', PROFILE['python'], spec['tag'], entry,
                '--data_path', str(data)]
        if a.check_mini_components:
            cmd += ['--output_dir', str(own_path(a.output_dir))]
        raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
