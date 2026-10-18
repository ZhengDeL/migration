"""Reload this model's checkpoint on NPU and evaluate its own held-out data."""
import argparse
from pathlib import Path
from common import ROOT, PROFILE, own_path, write_json, sha256, finite_tree
import model_task as task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--data_path', type=Path, required=True)
    args = parser.parse_args()
    output = own_path(args.output_dir, exists=True)
    data = own_path(args.data_path, exists=True)
    import torch
    import torch_npu
    torch.set_num_threads(2)
    torch_npu.npu.set_device(0)
    device = torch.device('npu', 0)
    evidence = __import__('json').loads((output / 'train_result.json').read_text())
    if sha256(output / 'checkpoint.pt') != evidence['checkpoint_sha256']:
        raise RuntimeError('checkpoint SHA256 与训练记录不同。')
    state = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    if state['model'] != PROFILE['model'] or sha256(data / 'metadata.json') != state['dataset_digest']:
        raise RuntimeError('checkpoint 的模型 / 数据摘要不匹配。')
    if not finite_tree(state) or state['step'] < 1 or not state['optimizer']['param_groups']:
        raise RuntimeError('checkpoint 状态、优化器或更新数无效。')
    if PROFILE.get('optimizer_has_tensor_state', True) and not state['optimizer']['state']:
        raise RuntimeError('有状态优化器的 tensor state 缺失。')
    task.check_data(data, 'full' if evidence['mode'] == 'full' else 'functional')
    from npu_precision import checkpoint_precision
    precision = checkpoint_precision(torch_npu, state['settings'])
    from npu_compat import restore_backend
    backend = restore_backend(state['settings'])
    model = task.create_model(state['metadata']).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    saved = torch.load(output / 'reload_reference.pt', map_location='cpu', weights_only=True)
    with torch.no_grad():
        actual = task.predict(model, saved['batch'], device).detach().cpu()
    torch.testing.assert_close(actual, saved['prediction'], rtol=1e-4, atol=1e-4)
    # Same stored checkpoint and input, independently reconstructed on CPU.
    cpu = task.create_model(state['metadata']).cpu()
    cpu.load_state_dict(state['state_dict'], strict=True)
    cpu.eval()
    with torch.no_grad():
        expected = task.predict(cpu, saved['batch'], torch.device('cpu')).detach()
    max_error = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    _, dataset = task.datasets(data, state['metadata'])
    metrics = task.evaluate(model, dataset, device, state['settings']['batch_size'], state['metadata'])
    result = {'model': PROFILE['model'], 'result': 'pass_checkpoint_reload_and_validation',
        'checkpoint_sha256': evidence['checkpoint_sha256'], 'optimizer_steps': state['step'],
        'strict_state_reload': True, 'reference_replay_matches': True,
        'cpu_npu_max_absolute_error': max_error, 'cpu_npu_rtol': 0.003, 'cpu_npu_atol': 0.003,
        'data_scope': state['metadata']['scope'], 'metrics': metrics, 'formal_accuracy_claimed': False}
    result['npu_precision'] = precision
    result['legacy_runtime_default_precision'] = 'npu_precision' not in state['settings']
    result['msda_backend'] = backend
    from npu_activations import activation_profile, activation_execution_counts
    result['activation_profile'] = activation_profile()
    result['source_checkpoint_activation_profile'] = state['settings'].get('activation_profile')
    result['activation_execution_counts'] = activation_execution_counts()
    from temporal_residual import TEMPORAL_RESIDUAL_PROFILE
    result['temporal_residual_profile'] = TEMPORAL_RESIDUAL_PROFILE
    result['source_checkpoint_temporal_residual_profile'] = state['settings'].get('temporal_residual_profile')
    from camera_geometry import CAMERA_PROJECTION_PROFILE, projection_execution_counts
    result['camera_projection_profile'] = CAMERA_PROJECTION_PROFILE
    result['source_checkpoint_camera_projection_profile'] = state['settings'].get('camera_projection_profile')
    result['projection_execution_counts'] = projection_execution_counts()
    if projection_execution_counts()['npu_pipeline'] == 0:
        raise RuntimeError('The constant camera projection did not execute in the NPU pipeline.')
    result['legacy_drivingsdk_autograd_profile'] = 'msda_backend' not in state['settings']
    write_json(output / 'validation_result.json', result)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
