"""Original FP32 camera projection on constant metadata, before NPU sampling."""
import torch

CAMERA_PROJECTION_PROFILE = 'constant_metadata_cpu_fp32_v1'
_calls = {'npu_pipeline': 0, 'cpu_reference': 0}


def projection_execution_counts():
    return dict(_calls)


def constant_sampling_grid(project, coor, sensor2ego, ego2global, cam2imgs,
                           post_rots, post_trans, bda, width, depth, device):
    """Run the pinned original projection and normalization on CPU FP32.

    Only fixed BEV coordinates and camera/augmentation metadata enter here.
    The original projection method is retained. Learned image features,
    sampling and their backward remain on the feature device.
    """
    values = (coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda)
    if any(not torch.is_tensor(value) or value.requires_grad for value in values):
        raise RuntimeError('Camera projection accepts only constant tensor metadata; learned inputs must stay on NPU.')
    if width < 1 or depth < 1:
        raise RuntimeError('Invalid original feature dimensions.')
    device = torch.device(device)
    if device.type not in ['cpu', 'npu']:
        raise RuntimeError('This validated projection profile supports CPU reference and NPU execution.')
    with torch.no_grad():
        metadata = [value.to(device='cpu', dtype=torch.float32) for value in values]
        projected = project(*metadata)
        if projected.device.type != 'cpu' or projected.dtype != torch.float32 or projected.requires_grad:
            raise RuntimeError('The original constant camera projection must return CPU FP32 coordinates.')
        if projected.ndim != 5 or projected.shape[-1] != 2:
            raise RuntimeError('The pinned camera projection output layout differs.')
        batch, cameras, height, columns, _ = projected.shape
        grid = projected.view(batch * cameras, height, columns, 2)
        # Keep the original CPU FP32 operations and their order exactly.
        grid[..., 0] = grid[..., 0] / width * 2 - 1
        grid[..., 1] = grid[..., 1] / depth * 2 - 1
        result = grid.to(device=device, dtype=torch.float32)
    _calls['npu_pipeline' if device.type == 'npu' else 'cpu_reference'] += 1
    return result
