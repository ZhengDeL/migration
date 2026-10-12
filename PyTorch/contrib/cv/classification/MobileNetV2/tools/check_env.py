"""Verify the actual runtime and NPU computation for this model."""
import argparse
import json
import platform
import sys
from pathlib import Path
from common import PROFILE, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch_npu
    import torchvision
    info = {'model': PROFILE['model'], 'architecture': platform.machine(), 'python': sys.version.split()[0],
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'numpy': np.__version__, 'torchvision': torchvision.__version__}
    if info['torch'] != '2.7.1+cpu' or info['torch_npu'] != '2.7.1.post6':
        raise RuntimeError('Torch / TorchNPU 与固定镜像版本不匹配。')
    if not torch_npu.npu.is_available() or torch_npu.npu.device_count() != args.nproc:
        raise RuntimeError('NPU 不可用或可见卡数不匹配。')
    devices = []
    for index in range(args.nproc):
        torch_npu.npu.set_device(index)
        x = torch.ones((8, 8), device='npu:' + str(index))
        value = float((x @ x).sum().cpu())
        if value != 512:
            raise RuntimeError('NPU 矩阵计算结果不正确。')
        devices.append({'index': index, 'name': torch_npu.npu.get_device_name(index), 'matrix_sum': value})
    info.update(result='pass_runtime_and_npu', devices=devices)
    write_json(args.output, info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
