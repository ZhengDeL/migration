"""Compute the original temporal difference without subtracting large sums."""
import torch

TEMPORAL_RESIDUAL_PROFILE = 'difference_before_weighted_sum_v1'


def temporal_residual(selected, features, position, frames, batch_size):
    """Use S @ (F[t]-F[t+1]) == S @ F[t] - S @ F[t+1].

    The original head repeats the same selection weights and positional
    embedding for all frames. FP32 subtraction after two long reductions
    loses small temporal signals. This equivalent order is used on both CPU
    and NPU; the unmodified FP64 head is an independent numerical oracle.
    """
    if frames < 2 or features.shape[0] != frames * batch_size:
        raise RuntimeError('Unexpected temporal feature layout.')
    single = features.reshape(frames, batch_size, features.shape[1], features.shape[2])
    difference = single[:-1] - single[1:]
    residual = torch.einsum('bsi,tbic->tbsc', selected, difference)
    # Retain the positional autograd connection. The repeated positional
    # tensors are identical, so this component and its direct derivative
    # cancel exactly; no ground truth or CPU tensor is substituted.
    position_delta = position - position
    position_residual = torch.einsum('bsi,bic->bsc', selected, position_delta)
    position_residual = position_residual.unsqueeze(0).expand(frames - 1, -1, -1, -1)
    return torch.cat((residual, position_residual), dim=-1)
