"""Exercise the pinned GeoBEV/SyncBN/EMA/warmup protocol on real mini inputs.

This bounded test does not enable or certify full trainval training.
"""
import argparse
import datetime
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

from common import ROOT, own_path, sha256, write_json, finite_tree, parameter_sha, tensor_tree_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, default=ROOT / 'data/nuscenes')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.steps <= 32:
        parser.error('This mini preflight accepts only 2..32 updates.')
    data = own_path(args.data_path, exists=True)
    output = own_path(args.output_dir, exists=True)
    from deterministic_communication import configure, attach, execution_counts
    communication = configure()
    import numpy as np
    import torch
    import torch_npu
    import torch.distributed as dist
    import model_task as task
    from npu_precision import configure_fp32
    from npu_compat import configure_training_backend
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise RuntimeError('The original-protocol preflight requires four distinct physical cards.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    precision = configure_fp32(torch_npu)
    torch_npu.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=300))
    metadata = task.check_data(data, 'functional')
    cfg = task.config()
    from mmcv.runner import load_checkpoint
    from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook
    from projects.mmdet3d_plugin.resworld.hooks.ema import ModelEMA
    hooks = {entry['type']: entry for entry in cfg.custom_hooks}
    ema_settings = hooks['MEGVIIEMAHook']
    if (cfg.total_epochs != 12 or hooks['SyncbnControlHook']['syncbn_start_epoch'] != 0
            or ema_settings['init_updates'] != 10560 or cfg.data.samples_per_gpu != 2):
        raise RuntimeError('Pinned original training protocol differs; review before execution.')
    warmup = dict(cfg.lr_config)
    if warmup != dict(policy='step', warmup='linear', warmup_iters=500,
                      warmup_ratio=1.0 / 3, step=[24]):
        raise RuntimeError('Pinned original learning-rate protocol differs.')
    torch.manual_seed(42)
    np.random.seed(42)
    task.initialize()
    backend = configure_training_backend()
    model = task.create_model(metadata)
    checkpoint_path = own_path(ROOT / cfg.load_from, exists=True)
    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained = checkpoint.get('state_dict', checkpoint)
    pretrained = {key.removeprefix('module.'): value for key, value in pretrained.items()}
    original_state = model.state_dict()
    matched = {key: value for key, value in pretrained.items()
               if key in original_state and value.shape == original_state[key].shape}
    if not matched or not any(key.startswith('img_backbone.') for key in matched):
        raise RuntimeError('GeoBEV checkpoint has no compatible image-backbone parameters.')
    # Use the original MMCV loader, including its documented strict=False mapping.
    load_checkpoint(model, str(checkpoint_path), map_location='cpu', strict=False)
    after_load = model.state_dict()
    if any(not torch.equal(after_load[key], value) for key, value in matched.items()):
        raise RuntimeError('The original loader did not restore every compatible GeoBEV tensor.')
    missing = sorted(set(original_state) - set(matched))
    unexpected = sorted(set(pretrained) - set(original_state))
    matched_count = len(matched)
    del checkpoint, pretrained, original_state, after_load, matched
    # This is the conversion called by upstream SyncbnControlHook at epoch zero.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None).to(device)
    bn_modules = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, torch.nn.SyncBatchNorm)]
    if not bn_modules or any(p.device.type != 'npu' for p in model.parameters()):
        raise RuntimeError('SyncBN conversion or NPU parameter placement failed.')
    calls = {'syncbn_forward': 0}
    def count_bn(module, inputs, result):
        calls['syncbn_forward'] += 1
        if result.device.type != 'npu' or not bool(torch.isfinite(result).all()):
            raise RuntimeError('SyncBN output is not a finite NPU tensor.')
    handles = [module.register_forward_hook(count_bn) for _, module in bn_modules]
    optimizer = torch.optim.AdamW(model.parameters(), **{key: cfg.optimizer[key]
        for key in ['lr', 'weight_decay']})
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    attach(wrapped)

    class EMAShell(torch.nn.Module):
        # The original ModelEMA expects the runner's parallel .module interface.
        def __init__(self, inner):
            super().__init__()
            self.module = inner
    # Match MEGVIIEMAHook's default, rather than ModelEMA's different default.
    ema = ModelEMA(EMAShell(model), decay=ema_settings.get('decay', 0.9990))
    ema.updates = ema_settings['init_updates']
    for handle in handles:
        handle.remove()
    # Count only actual training forwards, not the EMA copy's dormant hooks.
    handles = [module.register_forward_hook(count_bn) for _, module in bn_modules]
    lr_hook = StepLrUpdaterHook(**{key: value for key, value in warmup.items() if key != 'policy'})
    runner = SimpleNamespace(optimizer=optimizer, epoch=0, iter=0)
    lr_hook.before_run(runner)
    # Independently check transition boundaries against the original MMCV hook.
    schedule = []
    for epoch, step in [(0, 0), (0, 499), (0, 500), (11, 10000), (24, 10000)]:
        runner.epoch, runner.iter = epoch, step
        lr_hook.before_train_epoch(runner)
        lr_hook.before_train_iter(runner)
        expected = float(cfg.optimizer.lr) * (0.1 if epoch >= 24 else 1.0)
        if step < 500:
            expected *= 1 / 3 + (2 / 3) * step / 500
        actual = optimizer.param_groups[0]['lr']
        if not math.isclose(actual, expected, rel_tol=1e-12):
            raise RuntimeError('MMCV warmup/step boundary differs from the pinned formula.')
        schedule.append({'epoch': epoch, 'step': step, 'lr': actual})
    runner.epoch, runner.iter = 0, 0
    lr_hook.before_train_epoch(runner)
    dataset, validation = task.datasets(data, metadata)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True, seed=42)
    loader = DataLoader(dataset, batch_size=cfg.data.samples_per_gpu, sampler=sampler,
                        num_workers=0, drop_last=True, collate_fn=task.collate)
    if len(loader) < args.steps:
        raise RuntimeError('Insufficient real mini batches for the requested preflight.')
    torch.manual_seed(42 + rank)
    torch_npu.npu.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    initial = parameter_sha(model)
    rows = []
    model.train()
    for step, batch in enumerate(loader):
        runner.iter = step
        lr_hook.before_train_iter(runner)
        optimizer.zero_grad(set_to_none=True)
        loss, detail = task.loss(wrapped, batch, device)
        if loss.device.type != 'npu' or not bool(torch.isfinite(loss)):
            raise RuntimeError('Training loss must remain finite on NPU.')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
            max_norm=cfg.optimizer_config.grad_clip.max_norm, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        ema.update(runner, model)
        row = {'step': step + 1, 'loss': float(loss.detach().cpu()),
               'gradient_norm_before_clip': float(gradient_norm.cpu()),
               'lr': optimizer.param_groups[0]['lr'], 'ema_updates': ema.updates, **detail}
        rows.append(row)
        print(json.dumps({'rank': rank, **row}), flush=True)
        if step + 1 == args.steps:
            break
    final = parameter_sha(model)
    if initial == final or not calls['syncbn_forward'] or ema.updates != 10560 + args.steps:
        raise RuntimeError('Parameters, SyncBN execution or EMA updates are missing.')
    if not all(finite_tree(value) for value in [model.state_dict(), ema.ema.state_dict(), optimizer.state_dict()]):
        raise RuntimeError('Training/EMA/optimizer state contains nonfinite values.')
    checksum = torch.tensor([ord(value) for value in final], dtype=torch.int32, device=device)
    hashes = [torch.empty_like(checksum) for _ in range(world)]
    dist.all_gather(hashes, checksum)
    if any(not torch.equal(checksum, other) for other in hashes):
        raise RuntimeError('Full parameter hashes differ between ranks.')
    write_json(output / f'rank_{rank}.json', {'rank': rank, 'records': rows,
        'syncbn_modules': len(bn_modules), **calls, 'communication': execution_counts(),
        'final_parameter_sha256': final, 'finite_training_ema_optimizer': True})
    dist.barrier()
    if rank == 0:
        saved = {'schema': 'resworld_original_protocol_preflight_v1', 'state_dict': tensor_tree_cpu(model.state_dict()),
                 'ema_state_dict': tensor_tree_cpu(ema.ema.state_dict()), 'ema_updates': ema.updates,
                 'optimizer': tensor_tree_cpu(optimizer.state_dict()), 'step': args.steps,
                 'full_training_completed': False, 'exact_resume_claimed': False}
        torch.save(saved, output / 'checkpoint.pt')
        reloaded = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
        for key in ['state_dict', 'ema_state_dict']:
            if any(not torch.equal(value, reloaded[key][name]) for name, value in saved[key].items()):
                raise RuntimeError('Saved/reloaded original or EMA state differs.')
        write_json(output / 'protocol_preflight.json', {
            'result': 'pass_geobev_syncbn_ema_warmup_four_card_mini_preflight',
            'world_size': world, 'batch_size_per_rank': cfg.data.samples_per_gpu,
            'optimizer_updates': args.steps, 'syncbn_modules': len(bn_modules),
            'checkpoint_sha256': sha256(output / 'checkpoint.pt'),
            'geobev_checkpoint_sha256': checkpoint_sha, 'geobev_matched_tensors': matched_count,
            'geobev_missing_model_keys': missing,
            'geobev_unexpected_checkpoint_keys': unexpected, 'ema_updates': ema.updates,
            'ema_decay': ema_settings.get('decay', 0.9990), 'schedule_boundary_checks': schedule,
            'npu_precision': precision, 'communication': communication, 'msda_backend': backend,
            'state_dict_and_ema_reload_exact': True, 'all_rank_parameters_equal': True,
            'full_training_completed': False, 'full_accuracy_validated': False,
            'exact_resume_claimed': False, 'data_scope': metadata['scope'],
            'dataset': 'existing real mini functional training frames; no full trainval claim',
            'protocol_reference': 'vendor/ResWorld/projects/configs/resworld/resworld_config.py',
            'rows_rank0': rows})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
