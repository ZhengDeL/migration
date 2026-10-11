"""Verify finite FP32 updates, DDP synchronization and checkpoint states in legacy runners."""
import hashlib
import json
from pathlib import Path
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, OptimizerHook


def finite_tensors(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tensors(v) for v in value)
    return True


@HOOKS.register_module()
class MigrationFiniteOptimizerHook(OptimizerHook):
    def __init__(self, model_name, dataset_scope, grad_clip=None, expected_iterations=10,
                 initialization_checkpoint=None):
        super().__init__(grad_clip=grad_clip)
        self.model_name = model_name
        self.dataset_scope = dataset_scope
        self.expected_iterations = expected_iterations
        self.losses = []
        self.gradient_counts = []
        self.raw_output_checks = 0
        self.raw_loss_checks = 0
        self.loss_hooks = []
        self.initialization_checkpoint = initialization_checkpoint
        self.initialization_matches = []

    def before_run(self, runner):
        model = runner.model.module
        if self.initialization_checkpoint:
            path = Path(self.initialization_checkpoint)
            if hashlib.sha256(path.read_bytes()).hexdigest() != '2a81595fcee9df0ef93e9f76d7e735752786802f8a907c0c06949afb03ecea7a':
                raise RuntimeError('Official initialization checkpoint checksum mismatch')
            initial = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
            current = model.state_dict()
            for prefix in ('img_backbone.', 'pts_bbox_head.transformer.encoder.'):
                candidates = [name for name in initial if name.startswith(prefix) and name in current
                              and initial[name].is_floating_point() and initial[name].numel() > 100]
                if not candidates:
                    raise RuntimeError('No initialization tensor matches prefix: ' + prefix)
                name = candidates[0]
                if not torch.equal(initial[name], current[name].detach().cpu()):
                    raise RuntimeError('Official initialization was not loaded: ' + name)
                self.initialization_matches.append(name)
            del initial, current
        if self.model_name == 'worldengine':
            if model.planning_head.reward_shaping:
                raise RuntimeError('Functional training requires the pinned imitation-only config')
            head = model.planning_head
            self.watched = head.heads['imi'][-1].weight
        elif self.model_name == 'apollo_visionnet':
            head = model.pts_bbox_head
            self.watched = head.query_embedding.weight
            # The Apollo head masks detection losses inside loss_single.
            # Check each original loss module before that masking happens.
            def checked_loss_module(module, inputs, result):
                if not finite_tensors(result):
                    raise RuntimeError('Non-finite raw Apollo loss module output')
                self.raw_loss_checks += 1
            for name in ('loss_cls', 'loss_bbox', 'loss_occupancy'):
                self.loss_hooks.append(getattr(head, name).register_forward_hook(checked_loss_module))
        else:
            raise RuntimeError('Unsupported legacy model')
        self.before = self.watched.detach().clone()
        forward = head.forward
        loss = head.loss

        def checked_forward(*args, **kwargs):
            result = forward(*args, **kwargs)
            if not finite_tensors(result):
                raise RuntimeError('Non-finite raw model head output')
            self.raw_output_checks += 1
            return result

        def checked_loss(*args, **kwargs):
            result = loss(*args, **kwargs)
            if not finite_tensors(result):
                raise RuntimeError('Non-finite raw loss before upstream masking')
            if self.model_name == 'worldengine':
                self.raw_loss_checks += 1
            return result

        head.forward = checked_forward
        head.loss = checked_loss

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite total training loss')
        runner.optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in runner.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('Missing or non-finite real-data gradients')
        if self.grad_clip:
            norm = torch.nn.utils.clip_grad_norm_(runner.model.parameters(), foreach=False, **self.grad_clip)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('Non-finite gradient norm')
            runner.log_buffer.update({'grad_norm': float(norm)}, runner.outputs['num_samples'])
        runner.optimizer.step()
        self.losses.append(float(loss.detach()))
        self.gradient_counts.append(len(gradients))

    def after_run(self, runner):
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != 4 or len(self.losses) != self.expected_iterations or self.raw_output_checks == 0 or self.raw_loss_checks == 0:
            raise RuntimeError('Incomplete four-rank functional training')
        if not bool((self.before != self.watched.detach()).any()):
            raise RuntimeError('Watched parameter did not change')
        values = [torch.empty_like(self.watched) for _ in range(world)]
        dist.all_gather(values, self.watched.detach().contiguous())
        if not all(torch.allclose(self.watched.detach(), value, atol=1e-6, rtol=1e-5) for value in values):
            raise RuntimeError('Watched parameters differ after real-data DDP updates')
        report = {'model': self.model_name, 'dataset_scope': self.dataset_scope,
                  'rank': rank, 'world_size': world, 'iterations': len(self.losses),
                  'losses': self.losses, 'gradient_tensors_per_iteration': self.gradient_counts,
                  'raw_output_checks': self.raw_output_checks, 'raw_loss_checked_before_masking': True,
                  'raw_loss_checks': self.raw_loss_checks,
                  'parameter_changed': True, 'parameter_sync': 'pass', 'result': 'pass_training_smoke'}
        report['initialization_tensor_matches'] = self.initialization_matches
        checkpoint_error = None
        if rank == 0:
            try:
                checkpoint = Path(runner.work_dir) / f'iter_{self.expected_iterations}.pth'
                if not checkpoint.is_file():
                    raise RuntimeError('Expected checkpoint missing')
                # This checkpoint was created by this local authorized training run.
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['meta']['iter'] != self.expected_iterations - 1 or not finite_tensors(saved['state_dict']):
                    raise RuntimeError('Checkpoint model state or iteration invalid')
                states = saved['optimizer']['state']
                if not states or not finite_tensors(states):
                    raise RuntimeError('Checkpoint optimizer missing or non-finite')
                steps = [float(state['step']) for state in states.values() if 'step' in state]
                if not steps or any(step != self.expected_iterations for step in steps):
                    raise RuntimeError('Optimizer update counts do not match expected iterations')
                report['checkpoint_meta_iter'] = saved['meta']['iter']  # SDK stores the zero-based index.
                report['optimizer_step_values'] = sorted(set(steps))
                report['checkpoint_bytes'] = checkpoint.stat().st_size
                digest = hashlib.sha256()
                with checkpoint.open('rb') as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                report['checkpoint_sha256'] = digest.hexdigest()
                report['model_tensors'] = len(saved['state_dict'])
                report['optimizer_states'] = len(states)
                report['checkpoint_reload'] = 'pass'
            except Exception as error:
                checkpoint_error = str(error)
        checkpoint_ok = torch.tensor([int(checkpoint_error is None)], device=self.watched.device, dtype=torch.int32)
        dist.broadcast(checkpoint_ok, src=0)
        if not bool(checkpoint_ok.item()):
            raise RuntimeError(checkpoint_error or 'Rank-zero checkpoint validation failed')
        output = Path(runner.work_dir) / f'training_check_rank{rank}.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
