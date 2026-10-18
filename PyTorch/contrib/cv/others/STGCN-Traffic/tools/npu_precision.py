"""Explicit FP32 numerical baseline for this model in the pinned TorchNPU image."""


def configure_fp32(torch_npu):
    torch_npu.npu.conv.allow_hf32=False
    torch_npu.npu.matmul.allow_hf32=False
    if hasattr(torch_npu.npu,'aclnn'):torch_npu.npu.aclnn.allow_hf32=False
    state={'conv_allow_hf32':torch_npu.npu.conv.allow_hf32,
           'matmul_allow_hf32':torch_npu.npu.matmul.allow_hf32}
    if any(state.values()):raise RuntimeError('Explicit FP32 configuration did not take effect.')
    return state


def checkpoint_precision(torch_npu,settings):
    expected=settings.get('npu_precision')
    if expected is None:
        # Historical checkpoints retain their original runtime-default replay.
        # They are not silently compared under a different execution precision.
        return {'conv_allow_hf32':torch_npu.npu.conv.allow_hf32,
                'matmul_allow_hf32':torch_npu.npu.matmul.allow_hf32}
    actual=configure_fp32(torch_npu)
    if actual!=expected:raise RuntimeError('Stored checkpoint precision differs from the selected runtime.')
    return actual
