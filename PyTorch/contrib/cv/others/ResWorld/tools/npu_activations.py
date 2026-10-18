"""Preserve the upstream erf GELU on the pinned TorchNPU runtime."""
import types

import torch
import torch.nn.functional as F

GELU_MODULES = (
    'pts_bbox_head.tokenlearner.mlp.gelu',
    'pts_bbox_head.res_tokenlearner.mlp.gelu',
    'pts_bbox_head.tokenfuser.mlp.gelu',
)
_calls = {name: 0 for name in GELU_MODULES}


def activation_profile():
    return {'gelu_none': 'torch_npu.npu_gelu_erf', 'scope': list(GELU_MODULES)}


def activation_execution_counts():
    return dict(_calls)


def install_exact_gelu(model):
    """Bind only ResWorld's three GELUs; retain CPU reference and state keys."""
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, torch.nn.GELU)}
    if set(modules) != set(GELU_MODULES):
        raise RuntimeError('The pinned ResWorld GELU module structure differs.')
    for name, module in modules.items():
        if module.approximate != 'none':
            raise RuntimeError('Upstream ResWorld requires the default erf GELU.')
        if getattr(module, '_resworld_erf_gelu', False):
            continue

        def forward(self, value, name=name):
            if value.device.type == 'npu':
                import torch_npu
                _calls[name] += 1
                # F.gelu on this image uses the tanh approximation even for
                # approximate='none'. The official custom API preserves erf,
                # including its native NPU backward. No CPU training fallback.
                return torch_npu.npu_gelu(value, approximate='none')
            return F.gelu(value, approximate=self.approximate)

        module.forward = types.MethodType(forward, module)
        module._resworld_erf_gelu = True


def require_same_numerical_profile(settings):
    from camera_geometry import CAMERA_PROJECTION_PROFILE
    if settings.get('camera_projection_profile') != CAMERA_PROJECTION_PROFILE:
        raise RuntimeError('Exact continuation requires the recorded constant camera projection profile; preserve historical runs and start a new run.')
    from temporal_residual import TEMPORAL_RESIDUAL_PROFILE
    if settings.get('temporal_residual_profile') != TEMPORAL_RESIDUAL_PROFILE:
        raise RuntimeError('Exact continuation requires the recorded stable temporal reduction; preserve historical runs and start a new run.')
    if settings.get('activation_profile') != activation_profile():
        raise RuntimeError(
            'Exact continuation requires the recorded erf GELU profile. '
            'Preserve historical native-GELU checkpoints and start a new run; '
            'changing the activation implementation cannot be called exact resume.')
