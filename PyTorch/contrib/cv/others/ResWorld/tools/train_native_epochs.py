"""Exercise complete real-mini epochs using the original MMCV runner and hooks.

The full-data entry stays gated separately. Checkpoints here support exact
epoch-boundary continuation, including the original EMA and every rank's RNG.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import random
from pathlib import Path

from common import ROOT, own_path, sha256, write_json, tensor_tree_cpu, finite_tree, parameter_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, choices=[1, 2], required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--workers', type=int, default=0, choices=[0, 8])
    a = p.parse_args()
    output = own_path(a.output_dir, exists=True)
    data = own_path(ROOT / 'data/nuscenes', exists=True)
    from deterministic_communication import configure, attach, execution_counts
    communication = configure()
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    import model_task as task
    from npu_precision import configure_fp32
    from npu_compat import configure_training_backend
    from training_state import capture_rng, restore_rng, batch_digest

    rank, local, world = [int(os.environ[name]) for name in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']]
    if world != 4:
        raise RuntimeError('This original-runner control requires four distinct physical cards.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    precision = configure_fp32(torch_npu)
    torch_npu.npu.set_device(local)
    dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=300))
    metadata = task.check_data(data, 'functional')
    cfg = task.config()
    from mmcv.device.npu import NPUDistributedDataParallel
    from mmcv.parallel import DataContainer
    from mmcv.runner import EpochBasedRunner, build_optimizer, load_checkpoint, get_dist_info
    from mmcv.runner.hooks import Hook, DistSamplerSeedHook, OptimizerHook
    from mmcv.utils import build_from_cfg
    from mmcv.runner import HOOKS
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from projects.mmdet3d_plugin.resworld.hooks.ema import MEGVIIEMAHook

    if cfg.total_epochs != 12 or cfg.data.samples_per_gpu != 2:
        raise RuntimeError('Review the changed pinned original training protocol.')
    cfg.data.train.update(data_root=str(data),
        ann_file=str(data / 'vad_nuscenes_infos_temporal_train.pkl'),
        map_classes=cfg.map_classes, map_fixed_ptsnum_per_line=20)
    # Use the unfiltered original dataset and its own DistributedGroupSampler.
    dataset = custom_build_dataset(cfg.data.train)
    if len(dataset) != metadata['train_frames'] or len(dataset) != 323:
        raise RuntimeError('The native mini control must retain all 323 training annotations.')
    generator = torch.Generator().manual_seed(910000 + rank)
    loader = build_dataloader(dataset, cfg.data.samples_per_gpu, a.workers,
        num_gpus=world, dist=True, seed=42,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler, generator=generator)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    backend = configure_training_backend()
    model = task.create_model(metadata).to(torch.device('npu', local))
    weight = ROOT / cfg.load_from
    lock = json.loads((ROOT / 'pretrained_sources.lock.json').read_text())['files'][weight.name]
    if sha256(weight) != lock['sha256'] or weight.stat().st_size != lock['bytes']:
        raise RuntimeError('GeoBEV initialization differs from the pinned downloaded artifact.')
    load_checkpoint(model, str(weight), map_location='cpu', strict=False)

    def unwrap(value):
        if isinstance(value, DataContainer):
            return unwrap(value.data)
        if isinstance(value, (list, tuple)):
            return [unwrap(item) for item in value]
        return value

    class TracedNPUParallel(NPUDistributedDataParallel):
        def train_step(self, data_batch, *args, **kwargs):
            self.last_bucket_layout = []
            selected = {key: unwrap(data_batch[key]) for key in ['img_inputs', 'can_bus',
                'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_depth']}
            self.last_input_sha256 = batch_digest(selected)
            return super().train_step(data_batch, *args, **kwargs)

    # Keep constructor-order buckets on every step, including a restarted process.
    # With find_unused_parameters=False DDP rebuilds buckets after the first step;
    # that restart-dependent reduction layout failed the bit-exact control.
    wrapped = TracedNPUParallel(model, device_ids=[local], broadcast_buffers=False,
                               find_unused_parameters=True)
    parameter_names = {id(value): name for name, value in wrapped.module.named_parameters()}

    def observe_bucket(bucket):
        names = [parameter_names[id(value)] for value in bucket.parameters()]
        wrapped.last_bucket_layout.append({'index': bucket.index(), 'elements': bucket.buffer().numel(),
            'parameter_order_sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()})

    attach(wrapped, observer=observe_bucket)
    optimizer = build_optimizer(wrapped, cfg.optimizer)
    logger = logging.getLogger('resworld_native_epoch_' + str(rank))
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    runner = EpochBasedRunner(model=wrapped, optimizer=optimizer, work_dir=str(output),
        logger=logger, max_epochs=a.epochs, meta={'seed': 42, 'scope': 'real_mini_native_epoch_control'})
    runner.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    runner.register_training_hooks(cfg.lr_config, OptimizerHook(**cfg.optimizer_config),
        dict(cfg.checkpoint_config, max_keep_ckpts=2),
        dict(interval=10, hooks=[dict(type='TextLoggerHook')]))
    runner.register_hook(DistSamplerSeedHook())
    profile = {'communication': communication, 'npu_precision': precision,
               'msda_backend': backend, 'workers': a.workers, 'world_size': world,
               'seed': 42, 'batch_size_per_rank': 2,
               'ddp_find_unused_parameters': True,
               'ddp_bucket_policy': 'constructor_order_each_step',
               'dataset_digest': sha256(data / 'metadata.json'),
               'source_lock_sha256': sha256(ROOT / 'sources.lock.json'),
               'geobev_sha256': lock['sha256']}
    resumed = None
    if a.resume:
        previous = own_path(a.resume, exists=True)
        resumed = torch.load(previous, map_location='cpu', weights_only=True)
        if resumed.get('schema') != 'resworld_native_exact_epoch_v1' or resumed['profile'] != profile:
            raise RuntimeError('Native resume requires identical saved source/data/device/worker settings.')
        if resumed['meta']['epoch'] >= a.epochs:
            raise RuntimeError('The target total epoch count must exceed the saved epoch count.')
        runner.resume(str(previous), resume_optimizer=True, map_location='cpu')

    class RestoredAuthorEMAHook(MEGVIIEMAHook):
        def before_run(self, native_runner):
            super().before_run(native_runner)
            if resumed is not None:
                native_runner.ema_model.ema.load_state_dict(resumed['ema_state_dict'], strict=True)
                native_runner.ema_model.updates = resumed['ema_updates']

    for entry in cfg.custom_hooks:
        settings = dict(entry)
        priority = settings.pop('priority', 'NORMAL')
        if settings['type'] == 'MEGVIIEMAHook':
            settings.pop('type')
            hook = RestoredAuthorEMAHook(**settings)
        else:
            hook = build_from_cfg(settings, HOOKS)
        runner.register_hook(hook, priority=priority)

    log_path = output / f'rank_{rank}.jsonl'

    class ExactEpochHook(Hook):
        def __init__(self):
            self.restore_pending = resumed is not None
            self.records = 0

        def before_train_epoch(self, native_runner):
            if self.restore_pending:
                # Run after all original epoch hooks, before constructing the iterator.
                restore_rng(resumed['rank_rng'][rank], torch_npu, local)
                generator.set_state(resumed['loader_generator_states'][rank])
                self.restore_pending = False

        def after_train_iter(self, native_runner):
            loss = native_runner.outputs['loss']
            gradients = [parameter.grad for parameter in native_runner.model.parameters()
                         if parameter.grad is not None]
            if loss.device.type != 'npu' or not bool(torch.isfinite(loss)) or not gradients:
                raise RuntimeError('Native original train_step did not produce a finite NPU loss/gradient.')
            if not bool(torch.stack([torch.isfinite(value).all() for value in gradients]).all()):
                raise RuntimeError('Native optimizer encountered a nonfinite gradient.')
            row = {'rank': rank, 'epoch': native_runner.epoch, 'batch_index': native_runner.inner_iter,
                   'optimizer_step': native_runner.iter + 1, 'loss': float(loss.detach().cpu()),
                   'lr': optimizer.param_groups[0]['lr'],
                   'input_sha256': native_runner.model.last_input_sha256,
                   'gradient_bucket_layout': native_runner.model.last_bucket_layout,
                   'ema_updates': native_runner.ema_model.updates}
            with log_path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            self.records += 1
            if rank == 0:
                print('NATIVE_STEP', json.dumps(row), flush=True)

        def after_train_epoch(self, native_runner):
            epoch = native_runner.epoch + 1
            actual_model = native_runner.model.module
            state = actual_model.state_dict()
            ema = native_runner.ema_model
            if not all(finite_tree(value) for value in [state, ema.ema.state_dict(), optimizer.state_dict()]):
                raise RuntimeError('Nonfinite native model, optimizer or EMA state at epoch boundary.')
            if ema.updates != 10560 + native_runner.iter:
                raise RuntimeError('Original EMA counter does not match all optimizer updates.')
            torch.save({'rng': capture_rng(torch_npu, local), 'loader_generator': generator.get_state()},
                       output / f'epoch_{epoch}_rank_{rank}_rng.pt')
            checksum = torch.tensor([ord(value) for value in parameter_sha(actual_model)],
                                    dtype=torch.int32, device=torch.device('npu', local))
            hashes = [torch.empty_like(checksum) for _ in range(world)]
            dist.all_gather(hashes, checksum)
            if any(not torch.equal(checksum, value) for value in hashes):
                raise RuntimeError('Native full parameter digests differ across ranks.')
            dist.barrier()
            if rank == 0:
                random_states = [torch.load(output / f'epoch_{epoch}_rank_{value}_rng.pt',
                                           map_location='cpu', weights_only=True) for value in range(world)]
                snapshot = {'schema': 'resworld_native_exact_epoch_v1', 'profile': profile,
                    'meta': {'epoch': epoch, 'iter': native_runner.iter},
                    'state_dict': tensor_tree_cpu(state), 'optimizer': tensor_tree_cpu(optimizer.state_dict()),
                    'ema_state_dict': tensor_tree_cpu(ema.ema.state_dict()), 'ema_updates': ema.updates,
                    'rank_rng': [value['rng'] for value in random_states],
                    'loader_generator_states': [value['loader_generator'] for value in random_states],
                    'dataset_frames': len(dataset), 'batches_per_epoch': len(loader),
                    'full_trainval_training_completed': False}
                target = output / f'exact_epoch_{epoch}.pth'
                temp = target.with_suffix('.writing')
                torch.save(snapshot, temp)
                temp.replace(target)
                write_json(output / f'epoch_{epoch}_snapshot.json', {
                    'epoch': epoch, 'total_optimizer_steps': native_runner.iter,
                    'ema_updates': ema.updates, 'checkpoint_sha256': sha256(target),
                    'all_rank_parameters_equal': True, 'model_ema_optimizer_finite': True})
            dist.barrier()

    trace = ExactEpochHook()
    runner.register_hook(trace, priority='LOWEST')
    torch.manual_seed(42 + rank)
    torch_npu.npu.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    runner.run([loader], cfg.workflow)
    if rank == 0:
        write_json(output / 'native_epoch_result.json', {
            'result': 'pass_original_mmcv_runner_real_mini_epochs',
            'world_size': world, 'training_frames': len(dataset), 'batches_per_epoch': len(loader),
            'total_epochs': runner.epoch, 'total_optimizer_steps': runner.iter,
            'updates_this_run': trace.records, 'ema_updates': runner.ema_model.updates,
            'resume_from': str(a.resume) if a.resume else None, 'profile': profile,
            'native_checkpoint': f'epoch_{runner.epoch}.pth',
            'native_ema_checkpoint': f'epoch_{runner.epoch}_ema.pth',
            'exact_checkpoint': f'exact_epoch_{runner.epoch}.pth',
            'communication_calls': execution_counts(),
            'full_trainval_training_completed': False, 'full_accuracy_validated': False})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
