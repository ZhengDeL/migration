#!/usr/bin/env python3
"""Run the scripts and assets belonging to this model directory only."""
import argparse
import datetime
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'model.json').read_text(encoding='utf-8'))
MODEL = 'bevformer'

def command(*args, **kwargs):
    return subprocess.run(list(args), check=True, **kwargs)

def image():
    spec = json.loads((ROOT / 'images.lock.json').read_text())['images'][PROFILE['image_role']]
    probe = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', spec['tag']], capture_output=True, text=True)
    if probe.returncode or probe.stdout.strip() != spec['image_id'] + '|arm64':
        raise RuntimeError('请先运行 bash test/setup.sh 导入本模型的固定镜像；不使用其他 image ID。')
    return spec

def latest():
    path = ROOT / 'test/output/latest_run.json'
    if not path.is_file():
        raise RuntimeError('还没有本模型的训练结果；先运行训练脚本。')
    data = json.loads(path.read_text())
    if data.get('state') != 'completed':
        raise RuntimeError('最近一次训练没有成功完成，不能用较早结果冒充本次验证。')
    output = (ROOT / data['output_dir']).resolve(strict=True)
    if not output.is_relative_to(ROOT):
        raise RuntimeError('训练结果不在本模型目录内。')
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'check', 'prepare', 'plan', 'functional', 'full', 'validate', 'evaluate-mini', 'train-mini'])
    parser.add_argument('--device_id', '--devices', dest='devices', help='宿主逻辑 NPU ID；训练选不同物理卡')
    parser.add_argument('--data_path', type=Path, help='本模型目录内的数据目录')
    parser.add_argument('--output_dir', type=Path, help='本模型内独立的训练输出目录；须尚不存在')
    parser.add_argument('--dataset_id', default='mllm_demo', help='仅图文模型的注册数据集名')
    parser.add_argument('--checkpoint-profile', choices=('delivered-mini', 'official-r101-24ep'),
                        default='official-r101-24ep', help='完整 mini 评测使用的固定权重身份')
    parser.add_argument('--checkpoint', type=Path, help='本模型完整 mini 训练生成的检查点')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('在有 Ascend 驱动和 Docker 的 ARM64 Linux 服务器执行本脚本。')
    import fcntl
    with (ROOT / '.model_operation.lock').open('a') as operation:
        try:
            fcntl.flock(operation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('本模型目录已有操作正在运行；不同模型目录可独立使用。')
        if args.action == 'setup':
            archive = ROOT / 'images' / PROFILE['image_archive']
            if not archive.is_file():
                raise RuntimeError('缺少本模型镜像归档：' + str(archive) + '；请取得本模型完整离线交付包。')
            spec = json.loads((ROOT / 'images.lock.json').read_text())['images'][PROFILE['image_role']]
            probe = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}|{{.Architecture}}', spec['tag']], capture_output=True, text=True)
            if probe.returncode:
                command(sys.executable, str(ROOT / 'tools/verify_package.py'), '--assets-only')
                command('docker', 'load', '--input', str(archive))
            elif probe.stdout.strip() != spec['image_id'] + '|arm64':
                raise RuntimeError('同名镜像 ID 不一致，请先核对现有用途。')
            image()
            args.action = 'prepare'
        spec = image()
        nproc = 1 if args.action in ('validate', 'evaluate-mini') else PROFILE['nproc']
        host_ids = (args.devices or ('0' if nproc == 1 else PROFILE['host_devices'])).split(',')
        if args.action in ('prepare', 'plan'):
            host_ids = []
        else:
            if len(host_ids) != nproc or any((not x.isdigit() for x in host_ids)) or len(set(host_ids)) != nproc:
                raise RuntimeError('设备数不匹配；训练需 ' + str(nproc) + ' 个 ID，验证需 1 个 ID。')
            if len({int(x) // 2 for x in host_ids}) != nproc:
                raise RuntimeError('1# 双芯片卡拓扑中，每张物理卡只选一个逻辑 ID。')
            if args.action in ('functional', 'full', 'validate', 'evaluate-mini', 'train-mini'):
                command(sys.executable, str(ROOT / 'tools/check_idle_cards.py'), ','.join(host_ids))
        data = (args.data_path or ROOT / PROFILE['data_dir']).resolve()
        if not data.is_relative_to(ROOT):
            raise RuntimeError('--data_path 必须位于本模型目录下；先按 README 放置或迁入数据。')
        if args.action in ('functional', 'full', 'plan', 'evaluate-mini', 'train-mini'):
            output = (args.output_dir or ROOT / 'test/output' / (args.action + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
            if not output.is_relative_to(ROOT) or output.exists():
                raise RuntimeError('输出目录须在本模型目录内且尚不存在。')
            output.mkdir(parents=True)
            if args.action not in ('plan', 'evaluate-mini'):
                (ROOT / 'test/output/latest_run.json').write_text(json.dumps({'model': MODEL, 'state': 'running', 'output_dir': output.relative_to(ROOT).as_posix()}, indent=2) + '\n')
        elif args.action == 'validate':
            output = args.output_dir.resolve(strict=True) if args.output_dir else latest()
        else:
            output = ROOT / 'test/output'
            output.mkdir(parents=True, exist_ok=True)
        docker = ['docker', 'run', '--rm', '--network', 'none' if args.action in ('prepare', 'plan') else 'host', '--memory', PROFILE['memory'], '--cpus', '8', '--pids-limit', '1024', '--shm-size', '8g', '-v', str(ROOT) + ':' + str(ROOT) + ':rw', '-w', str(ROOT)]
        for item in ('/usr/local/Ascend/driver', '/usr/local/Ascend/firmware'):
            if Path(item).exists():
                docker += ['-v', item + ':' + item + ':ro']
        for item in ('/usr/local/bin/npu-smi', '/usr/local/dcmi', '/usr/bin/hccn_tool'):
            if Path(item).exists():
                docker += ['-v', item + ':' + item + ':ro']
        if host_ids:
            for item in ['/dev/davinci' + value for value in host_ids] + ['/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc']:
                if not Path(item).is_char_device():
                    raise RuntimeError('设备节点缺失：' + item)
                docker += ['--device', item]
            docker += ['-e', 'ASCEND_RT_VISIBLE_DEVICES=' + ','.join((str(i) for i in range(nproc)))]
        else:
            docker += ['-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0']
        docker += ['-e', 'OMP_NUM_THREADS=2', '-e', 'OPENBLAS_NUM_THREADS=2', '-e', 'PYTHONUNBUFFERED=1', '-e', 'HCCL_CONNECT_TIMEOUT=120', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'TOKENIZERS_PARALLELISM=false']
        if PROFILE.get('environment'):
            docker += ['-e', 'PATH=/opt/conda/envs/' + PROFILE['environment'] + '/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin']
        docker += ['--entrypoint', PROFILE['python'], spec['tag']]
        if args.action == 'prepare':
            launch = ['tools/prepare_data.py', '--data_path', str(data)]
        elif args.action == 'check':
            launch = ['tools/check_env.py', '--model', MODEL, '--nproc', str(nproc), '--output', 'test/output/environment.json', '--summary']
        elif args.action == 'evaluate-mini':
            launch = ['tools/evaluate_mini.py', '--output_dir', str(output),
                      '--checkpoint-profile', args.checkpoint_profile]
            if args.checkpoint:
                launch += ['--checkpoint', str(args.checkpoint.resolve(strict=True))]
        elif args.action == 'train-mini':
            launch = ['tools/train_mini.py', '--output_dir', str(output)]
        elif args.action == 'validate':
            launch = ['tools/validate_model.py', '--output_dir', str(output)]
        else:
            launch = ['tools/train_model.py', '--mode', 'full' if args.action == 'full' else 'functional', '--data_path', str(data), '--output_dir', str(output), '--dataset_id', args.dataset_id]
            if args.action == 'plan':
                launch += ['--plan']
        print('Model:', MODEL, '| image:', spec['tag'], '| host devices:', host_ids, flush=True)
        print('Model-only mount:', ROOT, flush=True)
        console = output / ('console_' + args.action + '.log')
        with console.open('wb') as transcript:
            child = subprocess.Popen(docker + launch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for block in iter(lambda : child.stdout.read1(65536), b''):
                transcript.write(block)
                transcript.flush()
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
            result_code = child.wait()
        if args.action == 'evaluate-mini':
            receipt = {'model': MODEL, 'action': args.action, 'image_tag': spec['tag'],
                'image_id': spec['image_id'], 'host_logical_devices': host_ids,
                'physical_cards': [int(value) // 2 for value in host_ids],
                'container_logical_devices': list(range(nproc)),
                'command': docker + launch, 'exit_code': result_code,
                'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (output / 'evaluation_execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if args.action == 'prepare' and result_code == 0:
            reference = ROOT / 'validation/reference_run'
            pointer = ROOT / 'test/output/latest_run.json'
            if reference.is_dir() and (not pointer.exists()):
                delivered = ROOT / 'test/output/delivered_reference'
                if delivered.exists():
                    raise RuntimeError('已有交付验证工作副本，保留后检查')
                shutil.copytree(reference, delivered)
                pointer.write_text(json.dumps({'model': MODEL, 'state': 'completed', 'mode': 'delivered_reference', 'image_id': spec['image_id'], 'exit_code': 0, 'output_dir': delivered.relative_to(ROOT).as_posix()}, indent=2) + '\n')
                print('Delivered reference copied to the model-local validation working directory.', flush=True)
        if args.action in ('functional', 'full', 'train-mini'):
            state = {'model': MODEL, 'state': 'completed' if result_code == 0 else 'failed', 'output_dir': output.relative_to(ROOT).as_posix(), 'exit_code': result_code, 'host_devices': host_ids, 'image_id': spec['image_id'], 'mode': args.action}
            (ROOT / 'test/output/latest_run.json').write_text(json.dumps(state, indent=2) + '\n')
        return result_code
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
