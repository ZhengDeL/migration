"""Complete PeMSD7(M) recipe: full epochs, validation selection, early stop and held-out test."""
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, PROFILE, own_path, parameter_sha, sha256, tensor_tree_cpu, finite_tree, write_json
import model_task as task


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def mse(model, dataset, device, batch_size):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = task.predict(model, batch, device)
            value = torch.nn.functional.mse_loss(prediction, batch[1].to(device))
            if not bool(torch.isfinite(value)):
                raise RuntimeError('Nonfinite complete-validation loss.')
            total += float(value.cpu()) * len(batch[0])
            count += len(batch[0])
    return total / count


def metric_report(model, dataset, metadata, device, batch_size):
    # Keep the original speed-domain MAE/RMSE/WMAPE definition from upstream.
    model.eval()
    mean = np.asarray(metadata['mean'], dtype=np.float64)
    std = np.asarray(metadata['std'], dtype=np.float64)
    absolute = squared = truth_sum = 0.0
    points = windows = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            prediction = task.predict(model, batch, device).cpu().numpy().astype(np.float64) * std + mean
            truth = batch[1].numpy().astype(np.float64) * std + mean
            if not np.isfinite(prediction).all():
                raise RuntimeError('Nonfinite complete-test prediction.')
            error = prediction - truth
            absolute += float(np.abs(error).sum())
            squared += float(np.square(error).sum())
            truth_sum += float(truth.sum())
            points += truth.size
            windows += len(truth)
    if truth_sum <= 0:
        raise RuntimeError('Upstream WMAPE requires a positive true speed sum.')
    return {'windows': windows, 'points': points, 'mae': absolute / points,
            'rmse': (squared / points) ** 0.5, 'wmape_ratio': absolute / truth_sum,
            'wmape_percent': 100 * absolute / truth_sum, 'forecast_minutes': 15,
            'units': 'original_speed_values'}


def main(args):
    data = own_path(args.data_path, exists=True)
    metadata = task.check_data(data, 'full')
    if not metadata['train_val_test_raw_intervals_disjoint']:
        raise RuntimeError('Full raw time intervals must be disjoint.')
    settings = dict(PROFILE['training']['full'])
    for key in ['epochs', 'lr']:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.steps not in [None, 0]:
        raise RuntimeError('Full training uses complete epochs and early stopping; use functional mode for step limits.')
    if settings['epochs'] < 1 or settings['lr'] <= 0:
        raise RuntimeError('epochs and lr must be positive.')
    if args.plan:
        print(json.dumps({'mode': 'full', 'scope': metadata['scope'], 'counts': metadata['counts'],
            'max_epochs': settings['epochs'], 'patience': 10, 'step_lr': {'step_size': 10, 'gamma': 0.95},
            'global_batch_size_default': 32, 'per_rank_batch_override': args.batch_size,
            'validation_selection': 'minimum normalized validation MSE',
            'deterministic_algorithms': True,
            'test_split_used_for_selection': False, 'npu_compute_executed': False}, indent=2))
        return
    import torch_npu
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    output = own_path(args.output_dir, exists=True)
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if 32 % world and args.batch_size is None:
        raise RuntimeError('Default global batch 32 must divide the rank count.')
    per_rank = args.batch_size if args.batch_size is not None else 32 // world
    if per_rank < 1:
        raise RuntimeError('Batch size must be positive.')
    global_batch = per_rank * world
    settings.update(batch_size=per_rank, global_batch_size=global_batch, early_stopping_patience=10, deterministic_algorithms=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    from npu_precision import configure_fp32
    settings['npu_precision']=configure_fp32(torch_npu)
    torch_npu.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if world > 1:
        dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=600))
    model = task.create_model(metadata).to(device)
    if any(p.device.type != 'npu' for p in model.parameters()):
        raise RuntimeError('Full-training parameters must all reside on NPU.')
    optimizer = task.create_optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    # Independent dropout streams after DDP has synchronized the same initial parameters.
    torch.manual_seed(42 + rank)
    torch_npu.npu.manual_seed(42 + rank)
    train, validation = task.datasets(data, metadata)
    with np.load(data / 'test.npz', allow_pickle=False) as archive:
        test = TensorDataset(torch.from_numpy(archive['x'].copy()), torch.from_numpy(archive['y'].copy()))
    dataset_digest = sha256(data / 'metadata.json')
    initial = parameter_sha(model)
    start_epoch = step = bad_epochs = 0
    best_loss, best_epoch = math.inf, 0
    history = []
    best_state = None
    if args.resume:
        state = torch.load(own_path(args.resume, exists=True), map_location='cpu', weights_only=True)
        if state.get('checkpoint_role') == 'validation_selected_best_model':
            raise RuntimeError('Use latest.pt for continuation; checkpoint.pt contains validation-selected model weights.')
        if state['model'] != PROFILE['model'] or state['dataset_digest'] != dataset_digest or state['world_size'] != world:
            raise RuntimeError('Resume model, full data or world size differs.')
        if state['settings']['global_batch_size'] != global_batch or state['settings']['lr'] != settings['lr'] or not state['settings'].get('deterministic_algorithms'):
            raise RuntimeError('Resume batch size or learning rate differs.')
        if state['settings'].get('npu_precision')!=settings['npu_precision']:
            raise RuntimeError('Full continuation requires identical recorded NPU precision; preserve the historical HF32 run and start fresh.')
        model.load_state_dict(state['state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, step = state['epochs_completed'], state['step']
        best_loss, best_epoch, bad_epochs = state['best_loss'], state['best_epoch'], state['bad_epochs']
        history, best_state = state['history'], state['best_state_dict']
        torch.set_rng_state(state['rank_rng'][rank]['torch'])
        torch_npu.npu.set_rng_state(state['rank_rng'][rank]['npu'], device=local_rank)
        initial = parameter_sha(model)
    stop_reason = 'max_epochs'
    for epoch in range(start_epoch, settings['epochs']):
        model.train()
        epoch_sum = epoch_count = 0
        begin = time.monotonic()
        for offset in range(0, len(train), global_batch):
            stop = min(offset + global_batch, len(train))
            indices = list(range(offset + rank, stop, world))
            real_count = len(indices)
            if not indices:
                indices = [offset]  # zero-weight filler only when a global remainder is smaller than world
            batch = (train.tensors[0][indices], train.tensors[1][indices])
            optimizer.zero_grad(set_to_none=True)
            prediction = wrapped(batch[0].to(device)).reshape(len(indices), -1)
            raw_loss = torch.nn.functional.mse_loss(prediction, batch[1].to(device))
            loss = raw_loss * (world * real_count / (stop - offset))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite full-training loss.')
            loss.backward()
            if step % 100 == 0:
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(g.device.type != 'npu' or not bool(torch.isfinite(g).all()) for g in gradients):
                    raise RuntimeError('Missing, non-NPU or nonfinite full-training gradient.')
            optimizer.step()
            step += 1
            epoch_sum += float(raw_loss.detach().cpu()) * real_count
            epoch_count += real_count
        totals = torch.tensor([epoch_sum, epoch_count], dtype=torch.float32, device=device)
        if world > 1:
            dist.all_reduce(totals)
        train_loss = float((totals[0] / totals[1]).cpu())
        if int(totals[1].cpu()) != len(train):
            raise RuntimeError('Full epoch must cover every training window exactly once.')
        scheduler.step()
        # A direct unwrapped rank-0 evaluation avoids DDP buffer collectives on only one rank.
        val_loss = mse(model, validation, device, 32) if rank == 0 else 0.0
        value = torch.tensor([val_loss], dtype=torch.float32, device=device)
        if world > 1:
            dist.broadcast(value, src=0)
        val_loss = float(value.cpu()[0])
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch + 1, 0
            best_state = tensor_tree_cpu(model.state_dict())
        else:
            bad_epochs += 1
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
               'lr_after_epoch': optimizer.param_groups[0]['lr'], 'bad_epochs': bad_epochs,
               'best_epoch': best_epoch, 'optimizer_steps': step,
               'training_windows_seen': int(totals[1].cpu()), 'seconds': time.monotonic() - begin}
        history.append(row)
        rng = {'torch': torch.get_rng_state(), 'npu': torch_npu.npu.get_rng_state(local_rank).cpu()}
        atomic_save(rng, output / ('rng_rank_' + str(rank) + '.pt'))
        if world > 1:
            dist.barrier()
        if rank == 0:
            checkpoint = {'model': PROFILE['model'], 'state_dict': tensor_tree_cpu(model.state_dict()),
                'optimizer': tensor_tree_cpu(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
                'step': step, 'epoch': epoch + 1, 'epochs_completed': epoch + 1,
                'world_size': world, 'dataset_digest': dataset_digest, 'metadata': metadata, 'settings': settings,
                'best_loss': best_loss, 'best_epoch': best_epoch, 'bad_epochs': bad_epochs,
                'best_state_dict': best_state, 'history': history,
                'rank_rng': [torch.load(output / ('rng_rank_' + str(r) + '.pt'), weights_only=True) for r in range(world)]}
            atomic_save(checkpoint, output / 'latest.pt')
            write_json(output / 'epoch_history.json', history)
            print(json.dumps(row), flush=True)
        if world > 1:
            dist.barrier()
        if args.pause_after_epochs and epoch + 1 >= args.pause_after_epochs:
            stop_reason = 'intentional_epoch_boundary_pause'
            break
        if bad_epochs >= 10:
            stop_reason = 'early_stopping'
            break
    if not history or best_state is None:
        raise RuntimeError('No complete training epoch / validation-selected checkpoint.')
    if parameter_sha(model) == initial or not finite_tree(model.state_dict()) or not finite_tree(optimizer.state_dict()):
        raise RuntimeError('Weights did not update or model / optimizer is nonfinite.')
    final = parameter_sha(model)
    if world > 1:
        value = torch.tensor(list(bytes.fromhex(final)), dtype=torch.int32, device=device)
        values = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(values, value)
        if any(not torch.equal(value, other) for other in values):
            raise RuntimeError('Full-training rank parameter SHA differs.')
    write_json(output / ('full_rank_' + str(rank) + '.json'), {'rank': rank, 'world_size': world,
        'epochs_completed': history[-1]['epoch'], 'optimizer_steps': step, 'parameters_on_npu': True,
        'parameter_sha256': final, 'state_finite': True, 'optimizer_finite': True,
        'initial_parameter_sha256': initial, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'all_training_windows_per_epoch': True, 'all_rank_parameter_hashes_match': True})
    if rank == 0:
        selected = torch.load(output / 'latest.pt', weights_only=True, map_location='cpu')
        selected = {key: selected[key] for key in ['model', 'world_size', 'dataset_digest', 'metadata', 'settings']}
        selected.update(state_dict=best_state, epoch=best_epoch,
            step=next(row['optimizer_steps'] for row in history if row['epoch'] == best_epoch),
            checkpoint_role='validation_selected_best_model', continuation_checkpoint='latest.pt')
        atomic_save(selected, output / 'checkpoint.pt')
        model.load_state_dict(selected['state_dict'], strict=True)
        sample = next(iter(DataLoader(validation, batch_size=32)))
        model.eval()
        with torch.no_grad():
            reference = task.predict(model, sample, device).cpu()
        cpu = task.create_model(metadata).cpu()
        cpu.load_state_dict(selected['state_dict'], strict=True)
        cpu.eval()
        with torch.no_grad():
            control = task.predict(cpu, sample, torch.device('cpu'))
        torch.testing.assert_close(reference, control, rtol=3e-3, atol=3e-3)
        atomic_save({'batch': sample, 'prediction': reference}, output / 'reload_reference.pt')
        result = {'model': PROFILE['model'], 'mode': 'full', 'result': 'complete_epoch_training_and_held_out_test_finished',
            'world_size': world, 'optimizer_steps': step, 'epochs_completed': history[-1]['epoch'],
            'stop_reason': stop_reason, 'best_epoch': best_epoch, 'best_validation_mse': best_loss,
            'dataset_digest': dataset_digest, 'data_scope': metadata['scope'], 'settings': settings,
            'data_path': data.relative_to(ROOT).as_posix(),
            'checkpoint_sha256': sha256(output / 'checkpoint.pt'), 'latest_checkpoint_sha256': sha256(output / 'latest.pt'),
            'cpu_npu_max_absolute_error': float((reference-control).abs().max()),
            'full_validation': metric_report(model, validation, metadata, device, 32),
            'held_out_test': metric_report(model, test, metadata, device, 32) if stop_reason != 'intentional_epoch_boundary_pause' else None,
            'full_reference_protocol_completed': stop_reason != 'intentional_epoch_boundary_pause' and settings['epochs'] == 1000 and global_batch == 32 and settings['lr'] == 0.001,
            'test_split_used_for_selection': False, 'formal_accuracy_claimed': False,
            'resume_from': str(args.resume) if args.resume else None}
        write_json(output / 'train_result.json', result)
        print(json.dumps(result, indent=2), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
