"""Check the pinned official Argoverse LSTM neural core on one NPU.

This is a synthetic coordinate feasibility check, not an Argoverse training
or evaluation pipeline. Only the two original neural classes are executed.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def original_classes(torch):
    lock = json.loads((ROOT / 'sources.lock.json').read_text(encoding='utf-8'))
    entry = next(item for item in lock['files'] if item['upstream_path'] == 'lstm_train_test.py')
    source = ROOT / entry['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise RuntimeError('The official source does not match sources.lock.json.')
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in ('EncoderRNN', 'DecoderRNN')]
    if [node.name for node in nodes] != ['EncoderRNN', 'DecoderRNN']:
        raise RuntimeError('Expected the two pinned official neural classes.')
    namespace = {'torch': torch, 'nn': torch.nn,
                 'F': torch.nn.functional, 'Any': Any}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['EncoderRNN'], namespace['DecoderRNN'], lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runtime_results/neural_core_result.json')
    args = parser.parse_args()
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch_npu.npu.set_device(0)
    EncoderRNN, DecoderRNN, lock = original_classes(torch)
    encoder, decoder = EncoderRNN(), DecoderRNN()
    cpu_encoder, cpu_decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
    encoder.to('npu:0')
    decoder.to('npu:0')
    history = torch.randn(2, 20, 2)
    target = torch.randn(2, 30, 2)

    def predict(enc, dec, inputs, device):
        inputs = inputs.to(device)
        state = (torch.zeros(2, 16, device=device),
                 torch.zeros(2, 16, device=device))
        for index in range(20):
            state = enc(inputs[:, index], state)
        current = inputs[:, -1]
        outputs = []
        for _ in range(30):
            current, state = dec(current, state)
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    actual = predict(encoder, decoder, history, 'npu:0')
    with torch.no_grad():
        expected = predict(cpu_encoder, cpu_decoder, history, 'cpu')
    if actual.device.type != 'npu' or not bool(torch.isfinite(actual).all()):
        raise RuntimeError('Predictions must be finite NPU tensors.')
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.001, atol=0.001)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    before = [parameter.detach().cpu().clone() for parameter in parameters]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    loss = torch.nn.functional.mse_loss(actual, target.to('npu:0'))
    if loss.device.type != 'npu' or not bool(torch.isfinite(loss)):
        raise RuntimeError('Loss must be a finite NPU tensor.')
    loss.backward()
    for parameter in parameters:
        if (parameter.device.type != 'npu' or parameter.grad is None
                or parameter.grad.device.type != 'npu'
                or not bool(torch.isfinite(parameter.grad).all())):
            raise RuntimeError('Every neural parameter must have finite NPU gradients.')
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise RuntimeError('Parameters must remain finite after the optimizer update.')
    changed = sum(not torch.equal(old, parameter.detach().cpu())
                  for old, parameter in zip(before, parameters))
    if changed == 0:
        raise RuntimeError('No neural parameters changed.')
    report = {
        'model': 'Argoverse LSTM trajectory baseline',
        'result': 'pass_original_neural_core_npu_feasibility',
        'source_revision': lock['revision'],
        'source_sha256': next(item['sha256'] for item in lock['files']
                              if item['upstream_path'] == 'lstm_train_test.py'),
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input_shape': list(history.shape), 'output_shape': list(actual.shape),
        'loss': float(loss.detach().cpu()),
        'cpu_npu_max_absolute_error': float((actual.detach().cpu() - expected).abs().max()),
        'comparison_rtol': 0.001, 'comparison_atol': 0.001,
        'parameter_tensors_updated': changed,
        'gradient_tensors_on_npu': len(parameters), 'optimizer_updates': 1,
        'seed': 42, 'dtype': 'float32', 'input_data': 'synthetic_coordinate_fixture',
        'torch_version': torch.__version__, 'torch_npu_version': torch_npu.__version__,
        'runtime_image_id': os.environ.get('MIGRATION_IMAGE_ID'),
        'real_argoverse_data_tested': False,
        'full_training_pipeline_verified': False,
        'distributed_training_verified': False, 'formal_accuracy_claimed': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
