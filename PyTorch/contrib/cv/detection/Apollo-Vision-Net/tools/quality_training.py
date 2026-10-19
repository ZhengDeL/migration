"""Record actual Apollo mini updates; fail on non-finite raw outputs, losses or gradients."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS
from finite_training import MigrationFiniteOptimizerHook, finite_tensors


@HOOKS.register_module()
class ApolloMiniQualityOptimizerHook(MigrationFiniteOptimizerHook):
    def after_train_iter(self, runner):
        super().after_train_iter(runner)
        rank = dist.get_rank()
        record = dict(epoch=runner.epoch + 1, update=runner.iter + 1,
            loss=self.losses[-1], gradient_tensors=self.gradient_counts[-1],
            raw_head_checks=self.raw_output_checks,
            log_vars={key: float(value) for key, value in runner.outputs['log_vars'].items()},
            lr=runner.current_lr(), collected_utc=datetime.now(timezone.utc).isoformat())
        with (Path(runner.work_dir) / f'quality_training_rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def after_run(self, runner):
        rank, world = dist.get_rank(), dist.get_world_size()
        expected = 24 * ((242 + 3) // 4)
        if world != 4 or len(self.losses) != expected or runner.epoch != 24:
            raise RuntimeError('Incomplete 24 epoch Apollo mini training')
        if not bool((self.before != self.watched.detach()).any()):
            raise RuntimeError('Training did not update watched parameters')
        values = [torch.empty_like(self.watched) for _ in range(world)]
        dist.all_gather(values, self.watched.detach().contiguous())
        if any(not torch.allclose(self.watched.detach(), value, atol=1e-6, rtol=1e-5) for value in values):
            raise RuntimeError('DDP watched parameters differ')
        report = dict(result='pass_completed_mini_training', model='apollo_visionnet',
            rank=rank, world_size=world, epochs=runner.epoch, updates=len(self.losses),
            train_frames=242, heldout_frames=162, train_val_disjoint=True,
            original_raw_loss_checks=self.raw_loss_checks, original_head_checks=self.raw_output_checks,
            parameter_changed=True, watched_parameter_sync=True,
            initialization_tensor_matches=self.initialization_matches,
            first_loss=self.losses[0], last_loss=self.losses[-1],
            task_quality_accepted=False, full_trainval_training_completed=False)
        failure = None
        if rank == 0:
            try:
                checkpoint = Path(runner.work_dir) / 'epoch_24.pth'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if not finite_tensors(saved['state_dict']) or not finite_tensors(saved['optimizer']):
                    raise RuntimeError('Checkpoint model or optimizer is non-finite')
                report.update(checkpoint=checkpoint.name,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    checkpoint_bytes=checkpoint.stat().st_size, model_tensors=len(saved['state_dict']),
                    optimizer_states=len(saved['optimizer']['state']))
            except Exception as error:
                failure = repr(error)
        good = torch.tensor([int(failure is None)], device=self.watched.device, dtype=torch.int32)
        dist.broadcast(good, src=0)
        if not good.item():
            raise RuntimeError(failure or 'Rank zero checkpoint verification failed')
        (Path(runner.work_dir) / f'quality_training_rank{rank}.json').write_text(
            json.dumps(report, indent=2, allow_nan=False) + '\n')
