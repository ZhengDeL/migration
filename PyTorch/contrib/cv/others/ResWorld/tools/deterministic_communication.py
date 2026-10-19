"""Fixed synchronous HCCL mean reduction for reproducible ResWorld training."""
import os

_calls = 0


def configure():
    # Set before the first HCCL communicator is constructed.
    os.environ['HCCL_DETERMINISTIC'] = 'true'
    return {'hccl_deterministic': True,
            'gradient_reduction': 'synchronous_mean_allreduce',
            'npu_synchronize_before_and_after': True}


def execution_counts():
    return {'synchronous_mean_allreduce': _calls}


def _mean_allreduce(state, bucket):
    import torch
    import torch_npu
    import torch.distributed as dist
    global _calls
    value = bucket.buffer()
    if value.device.type != 'npu':
        raise RuntimeError('ResWorld HCCL gradient buckets must stay on NPU.')
    if state is not None:
        state(bucket)
    torch_npu.npu.synchronize()
    value.div_(dist.get_world_size())
    dist.all_reduce(value, op=dist.ReduceOp.SUM, async_op=False)
    torch_npu.npu.synchronize()
    _calls += 1
    future = torch.futures.Future()
    future.set_result(value)
    return future


def attach(model, observer=None):
    """Use the public DDP communication-hook API; preserve mean-gradient math."""
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise RuntimeError('Attach the HCCL hook only to initialized multi-rank DDP.')
    model.register_comm_hook(observer, _mean_allreduce)
