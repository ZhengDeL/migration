"""Bounded real-data 2-NPU preflight; never labels this sixteen-row run full training."""

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/drivelm'
OUT = None
STATUS = None


def save(value):
    STATUS.write_text(json.dumps(value, indent=2) + '\n')


def main():
    started = time.time()
    global OUT, STATUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    from prepare_full_data import require_full_protocol, TRAIN_DATASET

    protocol = require_full_protocol(DATA, TRAIN_DATASET)
    from fetch_qwen import verify

    verify(ROOT / 'weights/qwen3-vl-8b-instruct')
    OUT = args.output_dir.resolve()
    if not OUT.is_relative_to(ROOT) or (OUT / 'trainer_state.json').exists() or any(OUT.glob('checkpoint-*')):
        raise RuntimeError('Use a fresh model-local output directory')
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS = OUT / 'preflight_result.json'
    state = {'state': 'preparing', 'complete_training': False, 'output_dir': str(OUT)}
    save(state)
    try:
        subsets = OUT / 'preflight_data'
        subsets.mkdir(exist_ok=True)
        selected = []
        seen = set()
        scene_order = []
        with (DATA / 'prepared/train.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                scene = row['scene_id']
                if scene not in scene_order:
                    if len(scene_order) == 4:
                        break
                    scene_order.append(scene)
                key = (scene, row['task'])
                if key not in seen:
                    selected.append(row)
                    seen.add(key)
        if len(selected) != 16 or len(seen) != 16:
            raise RuntimeError('Expected all four task categories from four separate training scenes')
        samples = subsets / 'qa.jsonl'
        samples.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
        registry = {
            'drivelm_preflight_16': {
                'file_name': 'qa.jsonl',
                'formatting': 'sharegpt',
                'columns': {'messages': 'messages', 'images': 'images'},
                'tags': {
                    'role_tag': 'role',
                    'content_tag': 'content',
                    'user_tag': 'user',
                    'assistant_tag': 'assistant',
                },
            }
        }
        (subsets / 'dataset_info.json').write_text(json.dumps(registry, indent=2))
        state.update(
            rows=16,
            scenes=4,
            qa_by_task=dict(collections.Counter(row['task'] for row in selected)),
            row_ids=[row['id'] for row in selected],
            source_file_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),
            state='training',
        )
        save(state)
        env = os.environ.copy()
        env.update(FORCE_TORCHRUN='1', NPROC_PER_NODE='2', PYTHONPATH=str(ROOT / 'sources/LLaMA-Factory/src'))
        base = [
            sys.executable,
            '-m',
            'torch.distributed.run',
            '--standalone',
            '--nnodes=1',
            '--nproc-per-node=2',
            str(ROOT / 'tools/drivelm_npu_entry.py'),
            str(ROOT / 'qwen3vl8b_drivelm_npu.yaml'),
            'model_name_or_path=' + str(ROOT / 'weights/qwen3-vl-8b-instruct'),
            'dataset_dir=' + str(subsets),
            'dataset=drivelm_preflight_16',
            'cutoff_len=' + str(protocol['cutoff_len']),
            'do_eval=false',
            "eval_strategy='no'",  # OmegaConf otherwise parses no as the boolean False.
            'eval_dataset=null',
            'packing=false',
            'logging_steps=1',
            'preprocessing_num_workers=2',
            'gradient_accumulation_steps=1',
            'max_steps=8',
            'num_train_epochs=1',
            'save_strategy=steps',
            'save_steps=8',
            'output_dir=' + str(OUT),
        ]
        state['train_command'] = base
        save(state)
        result = subprocess.run(base, env=env, cwd=ROOT / 'sources/LLaMA-Factory')
        state['train_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError('Real-data preflight training failed')
        command = [
            sys.executable,
            str(ROOT / 'tools/check_qwen_checkpoint.py'),
            '--output-dir',
            str(OUT),
            '--steps',
            '8',
            '--ranks',
            '2',
        ]
        subprocess.run(command, env=env, check=True)
        state['state'] = 'independent_reload_evaluation'
        save(state)
        evaluation = OUT / 'independent_eval'
        eval_command = [
            'llamafactory-cli',
            'train',
            str(ROOT / 'qwen3vl8b_drivelm_npu.yaml'),
            'model_name_or_path=' + str(ROOT / 'weights/qwen3-vl-8b-instruct'),
            'adapter_name_or_path=' + str(OUT),
            'dataset_dir=' + str(subsets),
            'dataset=null',
            'eval_dataset=drivelm_preflight_16',
            'do_train=false',
            'do_eval=true',
            'cutoff_len=' + str(protocol['cutoff_len']),
            'per_device_eval_batch_size=1',
            'prediction_loss_only=true',
            'output_dir=' + str(evaluation),
        ]
        state['eval_command'] = eval_command
        save(state)
        env.update(FORCE_TORCHRUN='0', NPROC_PER_NODE='1')
        result = subprocess.run(eval_command, env=env, cwd=ROOT / 'sources/LLaMA-Factory')
        state['eval_exit_code'] = result.returncode
        if result.returncode:
            raise RuntimeError('Independent adapter reload/evaluation failed')
        import math

        metrics = json.loads((evaluation / 'eval_results.json').read_text())
        if not math.isfinite(metrics['eval_loss']):
            raise RuntimeError('Independent real-data evaluation loss is nonfinite')
        state.update(
            state='pass_bounded_real_data_training_and_independent_reload',
            checkpoint=json.loads((OUT / 'checkpoint_check.json').read_text()),
            eval_metrics=metrics,
            official_accuracy_claimed=False,
        )
    except BaseException as error:
        state.update(state='failed', error=str(error))
        raise
    finally:
        state['seconds'] = time.time() - started
        save(state)


if __name__ == '__main__':
    main()
