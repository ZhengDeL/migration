"""Observe real BEV mini updates without changing its optimizer or raw outputs."""
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
class BEVMiniFiniteHook(Hook):
    def before_run(self, runner):
        model = runner.model.module
        self.initialization_matches = 0
        initial_path = Path(__file__).resolve().parents[1] / 'sources/BEVFormer/ckpts/r101_dcn_fcos3d_pretrain.pth'
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        initial = initial.get('state_dict', initial)
        live = model.state_dict()
        for name, tensor in initial.items():
            name = name.removeprefix('module.')
            if name.startswith('img_backbone.') and name in live:
                if live[name].shape != tensor.shape or not torch.equal(live[name].detach().cpu(), tensor.to(dtype=live[name].dtype)):
                    raise RuntimeError('Original FCOS3D backbone was not loaded exactly: ' + name)
                self.initialization_matches += 1
        if self.initialization_matches < 100:
            raise RuntimeError('Too few original backbone tensors matched')
        self.raw_checks = 0
        self.iterations = 0
        self.watched_name, self.watched = next((name, parameter)
            for name, parameter in model.pts_bbox_head.named_parameters() if parameter.requires_grad)
        self.before = self.watched.detach().clone()

        def checked(module, inputs, outputs):
            if not finite(outputs):
                raise RuntimeError('Non-finite raw BEVFormer head output')
            self.raw_checks += 1
        self.handle = model.pts_bbox_head.register_forward_hook(checked)

    def after_train_iter(self, runner):
        gradients = [parameter.grad for parameter in runner.model.parameters() if parameter.grad is not None]
        if not finite(runner.outputs['loss']) or not gradients or not finite(gradients):
            raise RuntimeError('Non-finite loss or missing/non-finite BEVFormer gradients')
        self.iterations += 1
        row = dict(update=runner.iter + 1, epoch=runner.epoch + 1,
            loss=float(runner.outputs['loss'].detach()), gradient_tensors=len(gradients),
            raw_checks=self.raw_checks, lr_unique=sorted({float(value) for value in runner.current_lr()}),
            log_vars={key: float(value) for key, value in runner.outputs['log_vars'].items()},
            collected_utc=datetime.now(timezone.utc).isoformat())
        with (Path(runner.work_dir) / f'quality_training_rank{dist.get_rank()}.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def after_run(self, runner):
        if self.iterations != 3888 or runner.epoch != 24 or not self.raw_checks:
            raise RuntimeError('Incomplete 24-epoch mini training')
        if not bool((self.before != self.watched.detach()).any()):
            raise RuntimeError('Watched BEV parameters did not change')
        report = dict(result='pass_completed_mini_training', model='bevformer', rank=dist.get_rank(),
            world_size=dist.get_world_size(), epochs=runner.epoch, updates=self.iterations,
            initialization_tensor_matches=self.initialization_matches, raw_checks=self.raw_checks,
            train_frames=323, heldout_frames=81, watched_parameter=self.watched_name,
            task_quality_accepted=False, full_trainval_training_completed=False)
        failure = None
        if dist.get_rank() == 0:
            try:
                checkpoint = Path(runner.work_dir) / 'epoch_24.pth'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if not finite(saved['state_dict']) or not finite(saved['optimizer']):
                    raise RuntimeError('Non-finite final model or optimizer')
                report.update(checkpoint=checkpoint.name, checkpoint_bytes=checkpoint.stat().st_size,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    model_tensors=len(saved['state_dict']), optimizer_states=len(saved['optimizer']['state']))
            except Exception as error:
                failure = repr(error)
        good = torch.tensor([int(failure is None)], device=self.watched.device, dtype=torch.int32)
        dist.broadcast(good, src=0)
        if not good.item():
            raise RuntimeError(failure or 'Rank zero checkpoint check failed')
        (Path(runner.work_dir) / f'quality_training_rank{dist.get_rank()}.json').write_text(
            json.dumps(report, indent=2) + '\n')
