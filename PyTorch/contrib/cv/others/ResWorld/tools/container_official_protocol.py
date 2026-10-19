"""Run the real four-card protocol preflight in this model's pinned image."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from common import ROOT, PROFILE, own_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,2,4,6')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    ids = args.devices.split(',')
    if len(ids) != 4 or any(not item.isdigit() for item in ids) or len({int(item) // 2 for item in ids}) != 4:
        parser.error('Select one logical device on each of four physical A3 cards.')
    output = own_path(args.output_dir)
    if output.exists():
        raise RuntimeError('Preserve earlier results; choose a new output directory.')
    spec = json.loads((ROOT / 'images.lock.json').read_text())['images']['runtime']
    actual = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', spec['tag']], text=True).strip()
    if actual != spec['image_id'] + '|arm64':
        raise RuntimeError('The runtime image differs from this model image lock.')
    import fcntl
    with (ROOT / '.model_operation.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([sys.executable, str(ROOT / 'tools/check_idle_cards.py'), args.devices], check=True)
        output.mkdir(parents=True)
        command = ['docker', 'run', '--rm', '--network', 'host', '--memory', '96g', '--cpus', '16',
                   '--shm-size', '8g', '-v', str(ROOT) + ':' + str(ROOT) + ':rw', '-w', str(ROOT)]
        for path in ['/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/dcmi']:
            if Path(path).exists():
                command += ['-v', path + ':' + path + ':ro']
        for path in ['/dev/davinci' + item for item in ids] + ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']:
            if not Path(path).is_char_device():
                raise RuntimeError('Missing device: ' + path)
            command += ['--device', path]
        for key, value in {'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3', 'OMP_NUM_THREADS': '2',
                           'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1',
                           'HCCL_CONNECT_TIMEOUT': '300',
                           'PATH': '/opt/conda/envs/bevformer/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'}.items():
            command += ['-e', key + '=' + value]
        command += ['--entrypoint', PROFILE['python'], spec['tag'], '-m', 'torch.distributed.run',
                    '--standalone', '--nproc-per-node=4', 'tools/preflight_official_protocol.py',
                    '--output_dir', str(output), '--steps', str(args.steps)]
        write_json(output / 'command.json', {'command': command, 'host_devices': ids,
                   'image_id': spec['image_id'], 'docker_host': os.environ.get('DOCKER_HOST', 'default'),
                   'full_training_completed': False})
        with (output / 'console.log').open('xb') as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        write_json(output / 'execution.json', {'exit_code': code, 'full_training_completed': False})
        if code:
            raise RuntimeError('Protocol preflight failed; inspect ' + str(output / 'console.log'))
        result = json.loads((output / 'protocol_preflight.json').read_text())
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
