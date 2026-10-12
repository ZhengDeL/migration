"""Process-local adaptations for the pinned training image and CPU reference."""
import torch


def install():
    import mx_driving
    import mmcv.ops.multi_scale_deform_attn as attention
    if getattr(attention.MultiScaleDeformableAttnFunction, '_resworld_npu', False):
        return
    original_attention = attention.MultiScaleDeformableAttnFunction.apply

    def dispatch(value, shapes, starts, locations, weights, im2col_step):
        if value.device.type == 'npu':
            return mx_driving.multi_scale_deformable_attn(value, shapes, starts, locations, weights)
        return original_attention(value, shapes, starts, locations, weights, im2col_step)

    attention.MultiScaleDeformableAttnFunction.apply = staticmethod(dispatch)
    attention.MultiScaleDeformableAttnFunction._resworld_npu = True
    # The image's Bottleneck calls an NPU fused op, including on CPU.
    original_add_relu = mx_driving.fused.npu_add_relu

    def add_relu(left, right):
        if left.device.type == 'cpu':
            return (left + right).relu()
        return original_add_relu(left, right)

    mx_driving.fused.npu_add_relu = add_relu


def restore_trainable_resnet_pool(backbone):
    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.relu(self.norm1(self.conv1(x)))
        # Upstream ResNet geometry: 3x3, stride 2, padding 1, ceil_mode False.
        # This image's inference-only replacement has no backward method.
        x = torch.ops.aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1], [1, 1], False)[0]
        outputs = []
        for index, name in enumerate(self.res_layers):
            x = getattr(self, name)(x)
            if index in self.out_indices:
                outputs.append(x)
        return tuple(outputs)

    backbone.forward = forward.__get__(backbone, type(backbone))
