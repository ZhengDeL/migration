"""Train this model with NPU and HCCL; save checkpoint and rank evidence."""
import argparse
import datetime
import json
import math
import os
import random
from pathlib import Path
from common import ROOT, PROFILE, own_path, write_json, sha256, finite_tree, parameter_sha, tensor_tree_cpu
import model_task as task
from training_state import capture_rng, restore_rng, epoch_loader, batch_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['functional', 'full'], required=True)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    data = own_path(args.data_path, exists=True)
    metadata = task.check_data(data, args.mode)
    settings = dict(PROFILE['training'][args.mode])
    for key in ['steps', 'epochs', 'batch_size', 'lr']:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if settings['epochs'] < 1 or settings['batch_size'] < 1 or settings['lr'] <= 0 or settings['steps'] < 0:
        raise RuntimeError('epochs / batch_size / lr 必须为正数，steps 不能为负数。')
    if args.plan:
        print(json.dumps({'model': PROFILE['model'], 'mode': args.mode, 'data': {k:v for k,v in metadata.items() if k not in ['files','actual_timestamp_intervals','mean','std']},
            'settings': settings, 'npu_compute_executed': False}, ensure_ascii=False, indent=2))
        return
    output = own_path(args.output_dir, exists=True)
    from deterministic_communication import configure, attach, execution_counts
    settings['communication'] = configure()
    import numpy as np
    import torch
    import torch_npu
    from npu_precision import configure_fp32
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    settings["deterministic_algorithms"] = True
    settings['npu_precision'] = configure_fp32(torch_npu)
    torch_npu.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    if world > 1:
        dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=180))
    task.initialize()
    from npu_compat import configure_training_backend, attention_execution_counts
    settings['msda_backend'] = configure_training_backend()
    from npu_activations import activation_profile, activation_execution_counts, require_same_numerical_profile
    settings['activation_profile'] = activation_profile()
    from temporal_residual import TEMPORAL_RESIDUAL_PROFILE
    settings['temporal_residual_profile'] = TEMPORAL_RESIDUAL_PROFILE
    from camera_geometry import CAMERA_PROJECTION_PROFILE, projection_execution_counts
    settings['camera_projection_profile'] = CAMERA_PROJECTION_PROFILE
    torch.manual_seed(42)
    np.random.seed(42)
    model = task.create_model(metadata).to(device)
    if any(p.device.type != 'npu' for p in model.parameters()):
        raise RuntimeError('模型参数未全部放在 NPU。')
    optimizer = task.create_optimizer(model, settings)
    dataset, validation = task.datasets(data, metadata)
    if hasattr(task, 'training_loader'):
        sampler, loader = task.training_loader(dataset, settings, rank, world)
    else:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=PROFILE.get('shuffle', True), drop_last=True)
        loader = DataLoader(dataset, batch_size=settings['batch_size'], sampler=sampler,
            num_workers=0, drop_last=True, collate_fn=task.collate)
    if not len(loader):
        raise RuntimeError('训练数据不足以组成每个 rank 的一个 batch。')
    wrapped = DistributedDataParallel(model, device_ids=[local_rank],
        find_unused_parameters=PROFILE.get('find_unused_parameters', False)) if world > 1 else model
    if world > 1:
        attach(wrapped)
    dataset_digest = sha256(data / 'metadata.json')
    torch.manual_seed(42 + rank)
    torch_npu.npu.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    step, start_epoch, start_batch = 0, 0, 0
    restored_complete_cursor = False
    if args.resume:
        previous = torch.load(own_path(args.resume, exists=True), map_location='cpu', weights_only=True)
        if previous['model'] != PROFILE['model'] or previous['dataset_digest'] != dataset_digest:
            raise RuntimeError('恢复权重的模型 / 数据摘要不同。')
        if previous['world_size'] != world:
            raise RuntimeError('继续训练时使用与保存 checkpoint 相同的 rank 数。')
        require_same_numerical_profile(previous['settings'])
        model.load_state_dict(previous['state_dict'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        step, start_epoch = previous['step'], previous['epoch']
        if previous.get('checkpoint_schema_version') == 2:
            if previous['settings'].get('communication') != settings['communication']:
                raise RuntimeError('Exact continuation requires the same recorded HCCL reduction profile; preserve the old run and start a new synchronized run.')
            if previous['settings'].get('msda_backend') != settings['msda_backend']:
                raise RuntimeError('Exact continuation requires the recorded attention backend; start a new reference-autograd run.')
            if previous['settings'].get('npu_precision') != settings['npu_precision']:
                raise RuntimeError('Exact continuation requires the same recorded NPU precision; preserve historical checkpoints and start a new FP32 run.')
            if previous['settings']['batch_size'] != settings['batch_size'] or previous['settings']['lr'] != settings['lr']:
                raise RuntimeError('Exact cursor continuation requires the original batch size and base learning rate.')
            start_epoch = previous['resume_cursor']['epoch']
            start_batch = previous['resume_cursor']['batch_index']
            restore_rng(previous['rank_rng'][rank], torch_npu, local_rank)
            restored_complete_cursor = True
    if settings['steps'] and step >= settings['steps']:
        raise RuntimeError('The target cumulative update count must exceed the saved update count.')
    initial = parameter_sha(model)
    records = []
    epoch = start_epoch
    cursor = {"epoch": start_epoch, "batch_index": start_batch}
    for epoch in range(start_epoch, settings['epochs']):
        sampler.set_epoch(epoch)
        model.train()
        if hasattr(task, 'begin_epoch'):
            task.begin_epoch()
        skip = start_batch if epoch == start_epoch else 0
        remaining = settings['steps'] - step if settings['steps'] else 0
        active_loader, total_epoch_batches = epoch_loader(loader, epoch, rank, skip, remaining)
        for batch_index, batch in enumerate(active_loader, start=skip):
            input_sha = batch_digest(batch) if args.mode == 'functional' else None
            optimizer.zero_grad(set_to_none=True)
            loss, detail = task.loss(wrapped, batch, device)
            if loss.device.type != 'npu' or not bool(torch.isfinite(loss)):
                raise RuntimeError('损失未在 NPU 或不是有限数。')
            loss.backward()
            if not all(activation_execution_counts().values()):
                raise RuntimeError('The expected native NPU erf GELUs did not all execute.')
            if projection_execution_counts()['npu_pipeline'] == 0:
                raise RuntimeError('The constant camera projection did not execute in the NPU pipeline.')
            if attention_execution_counts()['reference_autograd'] == 0:
                raise RuntimeError('The expected original MMCV NPU autograd path did not execute.')
            if world > 1 and execution_counts()['synchronous_mean_allreduce'] == 0:
                raise RuntimeError('The expected synchronous HCCL gradient reduction did not execute.')
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(g.device.type == 'npu' and bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError('梯度缺失或不是有限数。')
            gradient_norm = sum(float(g.detach().float().square().sum().cpu()) for g in gradients) ** 0.5
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError('梯度范数必须为有限正数。')
            if PROFILE.get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), PROFILE['clip_grad_norm'], foreach=False, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {'step': step, 'loss': float(loss.detach().cpu()), 'gradient_norm': gradient_norm,
                'gradient_tensors': len(gradients), 'epoch': epoch,
                'batch_index': batch_index, 'input_batch_sha256': input_sha, **detail}
            cursor = {'epoch': epoch + 1, 'batch_index': 0} if batch_index + 1 == total_epoch_batches else {'epoch': epoch, 'batch_index': batch_index + 1}
            records.append(row)
            print(json.dumps({'rank': rank, **row}), flush=True)
        if args.mode == 'full':
            for group in optimizer.param_groups:
                group['lr'] = settings['lr'] * PROFILE.get('epoch_lr_gamma', 1.0) ** ((epoch + 1) // 10)
        if settings['steps'] and step >= settings['steps']:
            break
    if not records:
        raise RuntimeError('没有执行优化器更新；--steps 表示目标累计更新数。')
    final = parameter_sha(model)
    if final == initial or not finite_tree(model.state_dict()) or not finite_tree(optimizer.state_dict()):
        raise RuntimeError('模型没有更新，或权重 / 优化器含非有限值。')
    if world > 1:
        checksum = torch.tensor([ord(c) for c in final], dtype=torch.int32, device=device)
        collected = [torch.empty_like(checksum) for _ in range(world)]
        dist.all_gather(collected, checksum)
        if any(not torch.equal(checksum, other) for other in collected):
            raise RuntimeError('各 rank 的完整参数摘要不同。')
        dist.barrier()
    model.eval()
    # Each rank retains its own real forward/backward evidence.
    write_json(output / ('rank_' + str(rank) + '.json'), {
        'rank': rank, 'world_size': world, 'steps_executed_this_run': len(records),
        'total_optimizer_steps': step, 'parameters_on_npu': True,
        'initial_parameter_sha256': initial, 'final_parameter_sha256': final,
        'parameters_updated': True, 'all_parameter_rank_hashes_match': True,
        'state_finite': True, 'optimizer_finite': True, 'records': records,
        'attention_execution_counts': attention_execution_counts(),
        'activation_execution_counts': activation_execution_counts(),
        'projection_execution_counts': projection_execution_counts(),
        'communication_execution_counts': execution_counts()})
    torch.save(capture_rng(torch_npu, local_rank), output / ('checkpoint_rng_rank_' + str(rank) + '.pt'))
    if world > 1:
        dist.barrier()
    if rank == 0:
        checkpoint = {'model': PROFILE['model'], 'state_dict': tensor_tree_cpu(model.state_dict()),
            'optimizer': tensor_tree_cpu(optimizer.state_dict()), 'step': step, 'epoch': epoch,
             'world_size': world, 'dataset_digest': dataset_digest, 'metadata': metadata, 'settings': settings,
            'checkpoint_schema_version': 2, 'resume_cursor': cursor,
            'rank_rng': [torch.load(output / ('checkpoint_rng_rank_' + str(r) + '.pt'), weights_only=True) for r in range(world)]}
        torch.save(checkpoint, output / 'checkpoint.pt')
        sample = next(iter(DataLoader(validation, batch_size=settings['batch_size'],
            num_workers=0, collate_fn=task.collate)))
        with torch.no_grad():
            reference = task.predict(model, sample, device).detach().cpu()
        if not finite_tree(reference):
            raise RuntimeError('保存前独立验证前向不是有限数。')
        torch.save({'batch': tensor_tree_cpu(sample), 'prediction': reference}, output / 'reload_reference.pt')
        write_json(output / 'train_result.json', {'model': PROFILE['model'], 'mode': args.mode,
            'result': 'pass_functional_training' if args.mode == 'functional' else 'training_completed',
            'world_size': world, 'optimizer_steps': step, 'dataset_digest': dataset_digest,
            'checkpoint_sha256': sha256(output / 'checkpoint.pt'), 'settings': settings,
            'data_scope': metadata['scope'], 'formal_accuracy_claimed': False,
            'resume_from': str(args.resume) if args.resume else None,
             'resume_semantics': 'all_rank_rng_and_epoch_batch_cursor_restored' if restored_complete_cursor else 'legacy_weight_optimizer_continuation' if args.resume else 'new_training',
            'bit_exact_replay_claimed': False, 'attention_execution_counts': attention_execution_counts(),
            'activation_execution_counts': activation_execution_counts(),
            'projection_execution_counts': projection_execution_counts(),
            'communication_execution_counts': execution_counts(),
            'resume_cursor': cursor})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
