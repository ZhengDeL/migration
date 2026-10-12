"""Launch only this independent model directory in its pinned ARM64 image."""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, PROFILE, own_path, write_json


def run(command, log=None):
    import time
    stamp = ROOT / 'test/output/last_container_exit.json'
    device_run = '--device' in command
    if device_run and stamp.is_file():
        remaining = 8 - (time.time() - json.loads(stamp.read_text())['finished_at'])
        if remaining > 0:
            print('Waiting for this model previous NPU container to release devices.', flush=True)
            time.sleep(remaining)
    print(' '.join(command), flush=True)
    if log is None:
        code = subprocess.run(command, check=False).returncode
    else:
        with Path(log).open('wb') as stream:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for block in iter(lambda: child.stdout.read1(65536), b''):
                stream.write(block)
                stream.flush()
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            code = child.wait()
    if device_run:
        write_json(stamp, {'finished_at': time.time(), 'exit_code': code})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'prepare', 'check', 'plan', 'functional', 'full', 'validate'])
    parser.add_argument('--nproc', type=int, choices=[1, 2], default=2)
    parser.add_argument('--devices', '--device_id', dest='devices')
    parser.add_argument('--data_path', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--mode', choices=['functional', 'full'], default='functional', help='plan only')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--imagenet_path', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('运行环境必须是 ARM64 Linux Ascend 服务器。')
    import fcntl
    with (ROOT / '.model_operation.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads((ROOT / 'images.lock.json').read_text())['images']['runtime']
        probe = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', spec['tag']], capture_output=True, text=True)
        if args.action == 'setup' and probe.returncode:
            subprocess.run([sys.executable, str(ROOT / 'tools/verify_package.py'), '--assets-only'], check=True)
            subprocess.run(['docker', 'load', '--input', str(ROOT / 'images' / spec['archive']['filename'])], check=True)
            probe = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', spec['tag']], capture_output=True, text=True)
        if probe.returncode or probe.stdout.strip() != spec['image_id'] + '|arm64':
            raise RuntimeError('镜像未导入或 image ID 不一致；按本模型 README 执行 setup。')
        if args.action == 'setup':
            args.action = 'prepare'
        mode = 'full' if args.action == 'full' else args.mode if args.action in ['plan', 'prepare'] else 'functional'
        pointer = ROOT / 'test/output/latest_run.json'
        if args.action == 'validate':
            args.nproc = 1
            if args.output_dir:
                output = own_path(args.output_dir, exists=True)
                state = json.loads((output / 'train_result.json').read_text())
                mode = state['mode']
                data_relative = state.get('data_path')
            else:
                state = json.loads(pointer.read_text())
                if state['state'] != 'completed':
                    raise RuntimeError('最近一次训练未完成，不能用更早结果冒充验证通过。')
                output = own_path(ROOT / state['output_dir'], exists=True)
                mode, data_relative = state['mode'], state['data_path']
            data = own_path(args.data_path or (ROOT / data_relative if data_relative else ROOT / PROFILE['data'][mode]), exists=True)
        else:
            data = own_path(args.data_path or ROOT / PROFILE['data'][mode])
            output = own_path(args.output_dir) if args.output_dir else ROOT / 'test/output' / (args.action + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.parent.mkdir(parents=True, exist_ok=True)
        ids = (args.devices or ','.join(PROFILE['host_devices'].split(',')[:args.nproc])).split(',')
        uses_npu = args.action not in ['prepare', 'plan']
        if uses_npu:
            if len(ids) != args.nproc or any(not item.isdigit() for item in ids) or len({int(v) // 2 for v in ids}) != args.nproc:
                raise RuntimeError('卡数不匹配；A3 每张物理卡选择一个逻辑 ID，例如双卡 0,2。')
            subprocess.run([sys.executable, str(ROOT / 'tools/check_idle_cards.py'), ','.join(ids)], check=True)
        docker = ['docker', 'run', '--rm', '--network', 'host' if uses_npu else 'none', '--memory', '48g', '--cpus', '8', '--pids-limit', '1024', '--shm-size', '8g', '-v', str(ROOT) + ':' + str(ROOT) + ':rw', '-w', str(ROOT)]
        for path in ['/usr/local/Ascend/driver', '/usr/local/Ascend/firmware', '/usr/local/bin/npu-smi', '/usr/local/dcmi']:
            if Path(path).exists():
                docker += ['-v', path + ':' + path + ':ro']
        if uses_npu:
            for path in ['/dev/davinci' + v for v in ids] + ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']:
                if not Path(path).is_char_device():
                    raise RuntimeError('设备节点不存在：' + path)
                docker += ['--device', path]
            docker += ['-e', 'ASCEND_RT_VISIBLE_DEVICES=' + ','.join(str(i) for i in range(args.nproc))]
        else:
            docker += ['-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0']
        for key, value in {'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2', 'PYTHONUNBUFFERED':'1', 'HCCL_CONNECT_TIMEOUT':'180',
            'PATH':'/opt/conda/envs/bevformer/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'}.items():
            docker += ['-e', key + '=' + value]
        docker += ['--entrypoint', PROFILE['python'], spec['tag']]
        options = ['--mode', mode, '--data_path', str(data)]
        for key in ['steps', 'epochs', 'batch_size', 'lr']:
            if getattr(args, key) is not None:
                options += ['--' + key, str(getattr(args, key))]
        if args.resume:
            options += ['--resume', str(own_path(args.resume, exists=True))]
        if args.action == 'prepare':
            if mode == 'full' and not args.imagenet_path:
                raise RuntimeError('Full data preparation requires --imagenet_path inside this model directory.')
            flags = ['--imagenet_path', str(own_path(args.imagenet_path, exists=True))] if args.imagenet_path else []
            return run(docker + ['tools/prepare_data.py'] + flags)
        if args.action == 'plan':
            return run(docker + ['tools/train.py', '--plan'] + options)
        if args.action == 'check':
            output.mkdir(exist_ok=False)
            return run(docker + ['tools/check_env.py', '--nproc', str(args.nproc), '--output', str(output / 'environment.json')], output / 'console.log')
        if args.action == 'validate':
            return run(docker + ['tools/validate.py', '--data_path', str(data), '--output_dir', str(output)], output / 'console_validate.log')
        # A data guard failure must preserve the previous completed training pointer.
        cpu_docker = docker.copy()
        code = run(cpu_docker + ['tools/train.py', '--plan'] + options)
        if code:
            return code
        if output.exists():
            raise RuntimeError('保留已有输出，请选择尚不存在的本模型输出目录。')
        output.mkdir()
        state = {'model': PROFILE['model'], 'mode': mode, 'data_path': data.relative_to(ROOT).as_posix(),
            'output_dir': output.relative_to(ROOT).as_posix(), 'image_id': spec['image_id'], 'host_devices': ids, 'nproc': args.nproc, 'state':'running'}
        write_json(pointer, state)
        command = docker + ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=' + str(args.nproc), 'tools/train.py'] + options + ['--output_dir', str(output)]
        code = run(command, output / 'console_train.log')
        state.update(state='completed' if code == 0 else 'failed', exit_code=code)
        write_json(pointer, state)
        if code == 0:
            result = json.loads((output / 'train_result.json').read_text())
            result['data_path'] = data.relative_to(ROOT).as_posix()
            write_json(output / 'train_result.json', result)
        return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, KeyError) as error:
        raise SystemExit(str(error))
