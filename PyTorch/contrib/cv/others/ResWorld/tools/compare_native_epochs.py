"""Compare native continuous/resumed model, EMA, optimizer, RNG and input traces."""
import argparse
import json
from pathlib import Path
from common import own_path, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resumed', type=Path, required=True)
    p.add_argument('--continuous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    resumed, continuous = [own_path(path, exists=True) for path in [a.resumed, a.continuous]]
    if resumed == continuous:
        raise RuntimeError('The resumed and independent continuous checkpoints must differ.')
    left = torch.load(resumed, map_location='cpu', weights_only=True)
    right = torch.load(continuous, map_location='cpu', weights_only=True)
    results = [json.loads((path.parent / 'native_epoch_result.json').read_text())
               for path in [resumed, continuous]]
    if not results[0]['resume_from'] or results[1]['resume_from'] is not None:
        raise RuntimeError('Require a restarted second epoch and an independent continuous two-epoch run.')
    for value, result in zip([left, right], results):
        if (value['meta'] != {'epoch': 2, 'iter': 82} or value['profile']['world_size'] != 4
                or value['dataset_frames'] != 323 or value['batches_per_epoch'] != 41
                or result['total_epochs'] != 2 or result['total_optimizer_steps'] != 82):
            raise RuntimeError('Require two complete native real-mini epochs on four ranks.')
    failures, tensor_count = [], 0

    def compare(x, y, path):
        nonlocal tensor_count
        if isinstance(x, torch.Tensor):
            tensor_count += 1
            if not isinstance(y, torch.Tensor) or x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                failures.append(path)
        elif isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                failures.append(path + ':keys')
                return
            for key in x:
                compare(x[key], y[key], path + '/' + str(key))
        elif isinstance(x, (list, tuple)):
            if type(x) != type(y) or len(x) != len(y):
                failures.append(path + ':length')
                return
            for index, (one, two) in enumerate(zip(x, y)):
                compare(one, two, path + '/' + str(index))
        elif x != y:
            failures.append(path)

    compare(left, right, 'checkpoint')
    trace_records = 0
    for rank in range(left['profile']['world_size']):
        first = [json.loads(line) for line in (resumed.parent / f'rank_{rank}.jsonl').read_text().splitlines()]
        second = [json.loads(line) for line in (continuous.parent / f'rank_{rank}.jsonl').read_text().splitlines()]
        for records, steps, epochs in [(first, list(range(42, 83)), {1}),
                                       (second, list(range(1, 83)), {0, 1})]:
            if ([row['optimizer_step'] for row in records] != steps
                    or {row['epoch'] for row in records} != epochs
                    or any(row['rank'] != rank for row in records)):
                raise RuntimeError('Missing, repeated or incorrectly assigned native trace records.')
        second = [row for row in second if row['epoch'] >= first[0]['epoch']]
        trace_records += len(first)
        compare(first, second, f'input_and_loss_trace/rank_{rank}')
    result = {'result': 'pass_native_epoch_resume_bit_exact' if not failures else 'fail_native_epoch_resume',
        'checkpoint_tensors_compared': tensor_count, 'trace_records_compared': trace_records,
        'mismatches': failures[:100], 'mismatch_count': len(failures),
        'resumed_checkpoint_sha256': sha256(resumed), 'continuous_checkpoint_sha256': sha256(continuous),
        'epoch': left['meta']['epoch'], 'optimizer_steps': left['meta']['iter'], 'ema_updates': left['ema_updates'],
        'complete_independent_control_inputs_validated': True,
        'full_trainval_training_completed': False, 'full_accuracy_validated': False}
    write_json(own_path(a.output), result)
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError('Native epoch continuation differs; preserve the comparison report.')


if __name__ == '__main__':
    main()
