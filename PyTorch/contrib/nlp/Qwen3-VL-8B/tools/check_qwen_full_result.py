#!/usr/bin/env python3
"""Check a completed three-epoch DriveLM run; do not infer official hidden scores."""
import json
import math
from pathlib import Path

from prepare_drivelm_complete import sha


def check(output):
    import torch
    from safetensors.torch import load_file

    output = Path(output)
    protocol = json.loads((output / 'full_data_verified.json').read_text())
    if protocol['train_examples'] != 340119 or protocol['eval_examples'] != 37836:
        raise RuntimeError('Unexpected completed-run data protocol')
    # 340119 rows, two ranks, per-rank batch one, accumulation eight, three epochs.
    expected = math.ceil(math.ceil(protocol['train_examples'] / 2) / 8) * 3
    state = json.loads((output / 'trainer_state.json').read_text())
    if state['global_step'] != expected or abs(float(state['epoch']) - 3.0) > 1e-6:
        raise RuntimeError('Three complete epochs have not finished')
    history = [row for row in state['log_history'] if 'loss' in row]
    if not history or not all(
        math.isfinite(float(row[key])) for row in history for key in ('loss', 'grad_norm')
    ):
        raise RuntimeError('Missing or nonfinite training loss/gradient records')
    checkpoint = output / ('checkpoint-' + str(expected))
    saved = json.loads((checkpoint / 'trainer_state.json').read_text())
    if saved['global_step'] != expected:
        raise RuntimeError('The final checkpoint trainer state is incomplete')
    adapter = output / 'adapter_model.safetensors'
    digest = sha(adapter)
    if digest != sha(checkpoint / adapter.name):
        raise RuntimeError('Final adapter and epoch-three checkpoint differ')
    values = load_file(str(adapter), device='cpu')
    if not values or not all(torch.isfinite(value).all().item() for value in values.values()):
        raise RuntimeError('Empty or nonfinite trained adapter')
    b = [value for name, value in values.items() if '.lora_B.' in name]
    if not b or not any(torch.count_nonzero(value).item() for value in b):
        raise RuntimeError('No trained LoRA B weights were found')
    optimizer = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)['state']
    if not optimizer or {int(value['step']) for value in optimizer.values()} != {expected}:
        raise RuntimeError('Final optimizer state does not match the completed updates')
    if not all(
        torch.isfinite(value[key]).all().item()
        for value in optimizer.values()
        for key in ('exp_avg', 'exp_avg_sq')
    ):
        raise RuntimeError('Nonfinite optimizer state')
    if not all((checkpoint / f'rng_state_{rank}.pth').is_file() for rank in range(2)):
        raise RuntimeError('Missing rank RNG states')
    evaluation = json.loads((output / 'eval_results.json').read_text())
    loss = float(evaluation['eval_loss'])
    if not math.isfinite(loss) or loss < 0:
        raise RuntimeError('Nonfinite or invalid complete local-dev loss')
    return {
        'result': 'pass_completed_three_epoch_training_and_local_dev_loss',
        'global_step': expected,
        'epochs': 3,
        'train_examples': 340119,
        'local_dev_examples': 37836,
        'rank_rng_artifacts': 2,
        'adapter_sha256': digest,
        'adapter_tensors': len(values),
        'optimizer_states': len(optimizer),
        'optimizer_step': expected,
        'recorded_training_loss_first': history[0]['loss'],
        'recorded_training_loss_last': history[-1]['loss'],
        'local_dev_eval_loss': loss,
        'exp_of_trainer_mean_eval_loss': math.exp(loss) if loss < 700 else None,
        'official_hidden_validation_score_computed': False,
        'independent_reload_evaluated': False,
    }
