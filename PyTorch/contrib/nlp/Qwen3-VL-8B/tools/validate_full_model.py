#!/usr/bin/env python3
"""Reload the completed DriveLM adapter and evaluate all local-dev rows on one NPU."""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from check_qwen_full_result import check
from prepare_full_data import DATA, ROOT, TRAIN_DATASET, require_full_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT):
        raise RuntimeError('结果必须在本模型目录中')
    checked = check(output)
    protocol = require_full_protocol(DATA, TRAIN_DATASET)
    from fetch_qwen import verify

    verify(ROOT / 'weights/qwen3-vl-8b-instruct')
    target = output / ('independent_full_eval_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    env = os.environ.copy()
    env.update(FORCE_TORCHRUN='0', NPROC_PER_NODE='1', PYTHONPATH=str(ROOT / 'sources/LLaMA-Factory/src'))
    command = [
        sys.executable,
        str(ROOT / 'tools/drivelm_npu_entry.py'),
        str(ROOT / 'qwen3vl8b_drivelm_npu.yaml'),
        'model_name_or_path=' + str(ROOT / 'weights/qwen3-vl-8b-instruct'),
        'adapter_name_or_path=' + str(output),
        'dataset_dir=' + protocol['dataset_dir'],
        'dataset=null',
        'eval_dataset=' + protocol['eval_dataset'],
        'do_train=false',
        'do_eval=true',
        'cutoff_len=' + str(protocol['cutoff_len']),
        'output_dir=' + str(target),
    ]
    subprocess.run(command, cwd=ROOT / 'sources/LLaMA-Factory', env=env, check=True)
    metrics = json.loads((target / 'eval_results.json').read_text())
    loss = float(metrics['eval_loss'])
    if not math.isfinite(loss) or not math.isclose(
        loss, checked['local_dev_eval_loss'], rel_tol=0.005, abs_tol=0.005
    ):
        raise RuntimeError('独立重载后的完整留出集 loss 未通过有限值/一致性检查')
    checked.update(
        result='pass_three_epoch_training_and_independent_complete_local_dev_reload',
        independent_reload_evaluated=True,
        independent_eval_examples=37836,
        independent_eval_loss=loss,
        independent_eval_dir=str(target),
        independent_eval_command=command,
        eval_loss_consistency_rtol=0.005,
        eval_loss_consistency_atol=0.005,
    )
    (output / 'full_validation_result.json').write_text(json.dumps(checked, indent=2) + '\n')
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    main()
