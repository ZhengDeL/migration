"""Run complete real-mini native epochs and their epoch-boundary continuation."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from common import ROOT, PROFILE, own_path, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,2,4,6')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, choices=[1, 2])
    parser.add_argument('--workers', type=int, choices=[0, 8], default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('RESUMED', 'CONTINUOUS'))
    args = parser.parse_args()
    if args.compare:
        if args.epochs or args.resume:
            parser.error('--compare is separate from training or continuation.')
        resumed, continuous = [own_path(path, exists=True) for path in args.compare]
        output = own_path(args.output_dir)
        if output.exists():
            raise RuntimeError('Preserve earlier comparisons; choose a new output directory.')
        spec = json.loads((ROOT / 'images.lock.json').read_text())['images']['runtime']
        actual = subprocess.check_output(['docker', 'image', 'inspect', '--format',
            '{{.Id}}|{{.Architecture}}', spec['tag']], text=True).strip()
        if actual != spec['image_id'] + '|arm64':
            raise RuntimeError('Pinned comparison image differs.')
        output.mkdir(parents=True)
        command = ['docker', 'run', '--rm', '--network', 'none', '--memory', '8g', '--cpus', '4',
            '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-v', str(ROOT) + ':' + str(ROOT) + ':rw', '-w', str(ROOT),
            '--entrypoint', PROFILE['python'], spec['tag'], 'tools/compare_native_epochs.py',
            '--resumed', str(resumed), '--continuous', str(continuous),
            '--output', str(output / 'native_resume_control.json')]
        write_json(output / 'command.json', {'command': command, 'image_id': spec['image_id'],
                   'npu_devices_exposed': False})
        with (output / 'console.log').open('xb') as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        write_json(output / 'execution.json', {'exit_code': code})
        if code:
            raise RuntimeError('Native resume comparison failed; inspect ' + str(output / 'console.log'))
        print((output / 'native_resume_control.json').read_text())
        return
    if args.epochs is None:
        parser.error('Training requires --epochs; a comparison requires --compare.')
    weight_name = 'geobev-r50-nuimage-cbgs.pth'
    weight_lock = json.loads((ROOT / 'pretrained_sources.lock.json').read_text())['files'][weight_name]
    weight = ROOT / 'ckpts' / weight_name
    if weight.stat().st_size != weight_lock['bytes'] or sha256(weight) != weight_lock['sha256']:
        raise RuntimeError('GeoBEV checkpoint differs from the model source lock.')
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
                    '--standalone', '--nproc-per-node=4', 'tools/train_native_epochs.py',
                    '--output_dir', str(output), '--epochs', str(args.epochs), '--workers', str(args.workers)]
        if args.resume:
            command += ['--resume', str(own_path(args.resume, exists=True))]
        write_json(output / 'command.json', {'command': command, 'host_devices': ids,
                   'image_id': spec['image_id'], 'docker_host': os.environ.get('DOCKER_HOST', 'default'),
                   'full_training_completed': False})
        with (output / 'console.log').open('xb') as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        write_json(output / 'execution.json', {'exit_code': code, 'full_training_completed': False})
        if code:
            raise RuntimeError('Native epoch run failed; inspect ' + str(output / 'console.log'))
        result = json.loads((output / 'native_epoch_result.json').read_text())
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
