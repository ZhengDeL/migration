"""Observe all real mini updates and reject raw non-finite Sparse4D outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


def finite(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


@HOOKS.register_module()
class SparseMiniFiniteHook(Hook):
    def before_run(self, runner):
        self.raw_checks = 0
        self.iterations = 0
        self.handles = []
        self.watched_name, self.watched = next((name, parameter)
            for name, parameter in runner.model.module.head.named_parameters() if parameter.requires_grad)
        self.before = self.watched.detach().clone()

        def checked(module, inputs, outputs):
            if not finite(outputs):
                raise RuntimeError('Non-finite raw Sparse4D head/keypoint output')
            self.raw_checks += 1
        for name, module in runner.model.module.named_modules():
            if name == 'head' or 'kps_generator' in name:
                self.handles.append(module.register_forward_hook(checked))

    def after_train_iter(self, runner):
        if not finite(runner.outputs['loss']):
            raise RuntimeError('Non-finite original training loss')
        gradients = [p.grad for p in runner.model.parameters() if p.grad is not None]
        if not gradients or not finite(gradients):
            raise RuntimeError('Missing/non-finite Sparse4D gradients after optimizer hook')
        self.iterations += 1
        row = dict(update=runner.iter + 1, loss=float(runner.outputs['loss'].detach()),
            gradient_tensors=len(gradients), raw_checks=self.raw_checks, lr=runner.current_lr(),
            log_vars={key: float(value) for key, value in runner.outputs['log_vars'].items()},
            collected_utc=datetime.now(timezone.utc).isoformat())
        with (Path(runner.work_dir) / f'quality_training_rank{dist.get_rank()}.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def after_run(self, runner):
        if self.iterations != 2600 or self.raw_checks == 0:
            raise RuntimeError('Incomplete original mini training schedule')
        if not bool((self.before != self.watched.detach()).any()):
            raise RuntimeError('Training did not update watched parameters')
        rank = dist.get_rank()
        report = dict(result='pass_completed_mini_training', model='sparse4d', rank=rank,
            world_size=dist.get_world_size(), updates=self.iterations, raw_checks=self.raw_checks,
            train_frames=323, heldout_frames=81, full_trainval_training_completed=False,
            task_quality_accepted=False, watched_parameter=self.watched_name)
        failure = None
        if rank == 0:
            try:
                path = Path(runner.work_dir) / 'iter_2600.pth'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if not finite(saved['state_dict']) or not finite(saved['optimizer']):
                    raise RuntimeError('Non-finite final model or optimizer state')
                report.update(checkpoint=path.name, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_bytes=path.stat().st_size, model_tensors=len(saved['state_dict']))
            except Exception as error:
                failure = repr(error)
        good = torch.tensor([int(failure is None)], device=self.watched.device, dtype=torch.int32)
        dist.broadcast(good, src=0)
        if not good.item():
            raise RuntimeError(failure or 'Rank zero checkpoint failed')
        (Path(runner.work_dir) / f'quality_training_rank{rank}.json').write_text(
            json.dumps(report, indent=2) + '\n')
