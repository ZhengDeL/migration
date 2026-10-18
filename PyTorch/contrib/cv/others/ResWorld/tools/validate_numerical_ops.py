"""Native NPU regression checks against the original formulas in CPU FP64."""
import argparse
from pathlib import Path

import torch
import torch_npu

from common import write_json
from temporal_residual import temporal_residual, TEMPORAL_RESIDUAL_PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch_npu.npu.set_device(0)
    torch_npu.npu.matmul.allow_hf32 = False
    torch_npu.npu.conv.allow_hf32 = False
    if hasattr(torch_npu.npu, 'aclnn'):
        torch_npu.npu.aclnn.allow_hf32 = False
    value = torch.cat((torch.linspace(-10, 10, 20001), torch.randn(100000)))
    upstream = torch.randn_like(value)
    reference_input = value.double().requires_grad_()
    reference = torch.nn.functional.gelu(reference_input, approximate='none')
    reference_grad = torch.autograd.grad(reference, reference_input, upstream.double())[0]
    actual_input = value.npu().requires_grad_()
    actual = torch_npu.npu_gelu(actual_input, approximate='none')
    actual_grad = torch.autograd.grad(actual, actual_input, upstream.npu())[0]
    torch.testing.assert_close(actual.cpu(), reference.float(), rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad.cpu(), reference_grad.float(), rtol=3e-5, atol=1e-6)
    native = torch.nn.functional.gelu(value.npu(), approximate='none').cpu()
    gelu = {'elements': value.numel(), 'official_api': 'torch_npu.npu_gelu(approximate=none)',
            'forward_max_abs_vs_fp64': float((actual.cpu().double() - reference).abs().max()),
            'backward_max_abs_vs_fp64': float((actual_grad.cpu().double() - reference_grad).abs().max()),
            'ordinary_functional_gelu_max_abs_vs_fp64': float((native.double() - reference).abs().max()),
            'forward_and_backward_on_npu': actual.device.type == actual_grad.device.type == 'npu'}

    frames, batch, length, channels, tokens = 3, 2, 17, 8, 4
    inputs = [torch.randn(batch, tokens, length), torch.randn(frames * batch, length, channels),
              torch.randn(batch, length, channels)]
    cpu_inputs = [v.double().requires_grad_() for v in inputs]
    selected, features, position = cpu_inputs
    original = torch.einsum('bsi,bic->bsc', selected.repeat(frames, 1, 1),
                           torch.cat((features, position.repeat(frames, 1, 1)), -1))
    original = original.reshape(frames, batch, tokens, channels * 2)
    original = original[:-1] - original[1:]
    upstream = torch.randn_like(original)
    expected_grad = torch.autograd.grad(original, cpu_inputs, upstream)
    npu_inputs = [v.npu().requires_grad_() for v in inputs]
    result = temporal_residual(*npu_inputs, frames, batch)
    gradients = torch.autograd.grad(result, npu_inputs, upstream.float().npu())
    torch.testing.assert_close(result.cpu(), original.float(), rtol=3e-5, atol=2e-6)
    for actual, expected in zip(gradients, expected_grad):
        if actual.device.type != 'npu':
            raise RuntimeError('Temporal backward must remain on NPU.')
        torch.testing.assert_close(actual.cpu(), expected.float(), rtol=3e-5, atol=2e-6)
    residual = {'profile': TEMPORAL_RESIDUAL_PROFILE,
                'oracle': 'unmodified separate weighted sums and subtraction, CPU float64',
                'forward_max_abs_vs_fp64': float((result.cpu().double() - original).abs().max()),
                'gradient_max_abs_vs_fp64': [float((a.cpu().double() - b).abs().max())
                                            for a, b in zip(gradients, expected_grad)],
                'gradient_inputs': ['selection_weights', 'temporal_features', 'repeated_position'],
                'forward_and_backward_on_npu': True}

    # Compare the retained, pinned original projection with the actual NPU
    # pipeline. The source projection method itself is not rewritten.
    import model_task as task
    task.initialize()
    from projects.mmdet3d_plugin.resworld.rcsample import RCSample
    from camera_geometry import constant_sampling_grid, CAMERA_PROJECTION_PROFILE
    dummy = torch.nn.Module()
    dummy.fp16_enabled = False
    dummy.downsamples = [1]
    dummy.grid_config = {'depth': [1, 60, 1]}
    def project(*values):
        return RCSample.get_sample_coor(dummy, *values)
    batch, cameras, height, width = 2, 6, 7, 5
    coordinates = torch.rand(batch, height, width, 3) + torch.tensor([1., 1., 4.])
    pose = torch.eye(4).repeat(batch, cameras, 1, 1)
    pose[..., :3, 3] = torch.randn(batch, cameras, 3) * 0.1
    camera = torch.eye(3).repeat(batch, cameras, 1, 1)
    camera[..., 0, 0] = camera[..., 1, 1] = 2
    rotations = torch.eye(3).repeat(batch, cameras, 1, 1)
    translations = torch.zeros(batch, cameras, 3)
    augmentation = torch.eye(3).repeat(batch, 1, 1)
    metadata = [coordinates, pose, pose.clone(), camera, rotations, translations, augmentation]
    expected = project(*metadata).view(batch * cameras, height, width, 2)
    expected[..., 0] = expected[..., 0] / 7 * 2 - 1
    expected[..., 1] = expected[..., 1] / 8 * 2 - 1
    grid = constant_sampling_grid(project, *[v.npu() for v in metadata], 7, 8, torch.device('npu:0'))
    if not torch.equal(grid.cpu(), expected):
        raise RuntimeError('The original CPU FP32 camera projection/normalization changed.')
    features = torch.randn(batch * cameras, 2, 8, 7).npu().requires_grad_()
    sampled = torch.nn.functional.grid_sample(features, grid, align_corners=False)
    sampled.square().sum().backward()
    if features.grad.device.type != 'npu' or not bool(torch.isfinite(features.grad).all()) or not bool(features.grad.abs().sum() > 0):
        raise RuntimeError('Neural feature sampling/backward did not remain active on NPU.')
    rejected = False
    try:
        learned = list(metadata)
        learned[0] = learned[0].clone().requires_grad_()
        constant_sampling_grid(project, *learned, 7, 8, torch.device('npu:0'))
    except RuntimeError as error:
        rejected = 'constant tensor metadata' in str(error)
    if not rejected:
        raise RuntimeError('Learned coordinate inputs must not be silently detached for CPU geometry.')
    geometry = {'profile': CAMERA_PROJECTION_PROFILE,
                'original_cpu_fp32_projection_and_normalization_bit_equal': True,
                'sampled_feature_gradient_device': features.grad.device.type,
                'sampled_feature_gradient_active_and_finite': True,
                'learned_metadata_rejected': rejected}
    write_json(args.output, {'result': 'pass_native_npu_numerical_operator_regression',
                            'scope': 'operator regression; real-model checks are recorded separately',
                            'gelu': gelu, 'temporal_residual': residual, 'constant_camera_geometry': geometry})


if __name__ == '__main__':
    main()
