#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Read-only Ascend training environment check for one selected model."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

def package(name: str, quiet: bool=False) -> tuple[object | None, str | None]:
    try:
        with warnings.catch_warnings(record=quiet):
            return (importlib.import_module(name), None)
    except Exception as exc:
        return (None, f'{type(exc).__name__}: {exc}')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('sparse4d',))
    parser.add_argument('--nproc', type=int, required=True)
    parser.add_argument('--output', type=Path, help='optional local JSON evidence file')
    parser.add_argument('--summary', action='store_true', help='print a short result; still save full JSON with --output')
    parser.add_argument('--require-npu-smi', action='store_true', help='also fail when the management CLI is unavailable inside the container')
    args = parser.parse_args()
    if args.nproc < 1:
        parser.error('--nproc must be positive')
    info: dict[str, object] = {'model': args.model, 'python': sys.version.split()[0], 'visible_devices': os.getenv('ASCEND_RT_VISIBLE_DEVICES'), 'cann_home': os.getenv('ASCEND_HOME_PATH'), 'checks': {}}
    checks: dict[str, object] = info['checks']
    (torch, checks['torch_import_error']) = package('torch', args.summary)
    (torch_npu, checks['torch_npu_import_error']) = package('torch_npu', args.summary)
    if torch is not None:
        info['torch'] = getattr(torch, '__version__', 'unknown')
    if torch_npu is not None:
        info['torch_npu'] = getattr(torch_npu, '__version__', 'unknown')
    if torch is not None and torch_npu is not None:
        torch_version = re.match('\\d+\\.\\d+', str(torch.__version__))
        npu_version = re.match('\\d+\\.\\d+', str(torch_npu.__version__))
        checks['torch_version_match'] = bool(torch_version and npu_version and (torch_version.group() == npu_version.group()))
        try:
            info['npu_available'] = bool(torch.npu.is_available())
            info['npu_count'] = int(torch.npu.device_count())
            info['devices'] = [torch.npu.get_device_name(i) for i in range(min(args.nproc, int(info['npu_count'])))]
            checks['enough_npus'] = bool(info['npu_available'] and int(info['npu_count']) >= args.nproc)
        except Exception as exc:
            checks['npu_query_error'] = f'{type(exc).__name__}: {exc}'
    dependencies = ('mmcv', 'mmdet')
    for dependency in dependencies:
        (module, error) = package(dependency, args.summary)
        checks[f'{dependency}_import_error'] = error
        if module is not None:
            info[f'{dependency}_version'] = getattr(module, '__version__', 'unknown')
    (mx_driving, checks['mx_driving_import_error']) = package('mx_driving', args.summary)
    if mx_driving is not None:
        checks['msda_operator'] = callable(getattr(mx_driving, 'multi_scale_deformable_attn', None))
    (mmcv_npu, checks['mmcv_npu_import_error']) = package('mmcv.device.npu', args.summary)
    if mmcv_npu is not None:
        checks['npu_distributed_wrapper'] = hasattr(mmcv_npu, 'NPUDistributedDataParallel')
    try:
        probe = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, timeout=10)
        checks['npu_smi_exit_code'] = probe.returncode
        checks['npu_smi_ok'] = probe.returncode == 0
        if probe.returncode:
            info['npu_smi_output'] = (probe.stdout + probe.stderr).strip()[-500:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks['npu_smi_error'] = str(exc)
    management_checks = {'npu_smi_ok', 'npu_smi_error'}
    failures = [name for (name, value) in checks.items() if name not in management_checks and (value is False or (name.endswith('_error') and value))]
    warnings = [name for name in management_checks if name in checks and (checks[name] is False or (name.endswith('_error') and checks[name]))]
    if args.require_npu_smi:
        failures.extend(warnings)
        warnings = []
    info['result'] = 'fail' if failures else 'pass_with_warnings' if warnings else 'pass'
    info['failures'] = failures
    info['warnings'] = warnings
    report = json.dumps(info, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    if args.summary:
        print(f"{args.model}: {info['result']} | NPU {info.get('npu_count', 0)}/{args.nproc} | JSON {args.output or '(not saved)'}")
        if failures:
            print(f"失败项: {', '.join(failures)}")
        if warnings:
            print(f"告警项: {', '.join(warnings)}；请以宿主机 npu-smi info 和实际计算交叉核对")
    else:
        print(report)
    return 0 if not failures else 2
if __name__ == '__main__':
    raise SystemExit(main())
