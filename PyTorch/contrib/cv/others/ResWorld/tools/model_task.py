"""Original ResWorld network, real nuScenes mini, independent planning validation."""
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from common import ROOT, sha256
from prepare_source import prepare_source

_initialized = False


def initialize():
    global _initialized
    if not _initialized:
        source = prepare_source()
        sys.path.insert(0, str(source))
        import projects.mmdet3d_plugin
        from npu_compat import install
        install()
        _initialized = True


def config():
    initialize()
    import mmcv
    return mmcv.Config.fromfile(str(ROOT / 'runtime/ResWorld/projects/configs/resworld/resworld_config.py'))


def check_data(path, mode):
    if mode == 'full':
        raise RuntimeError('Full-data GeoBEV initialization, EMA and official planning/collision acceptance are not yet validated.')
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata['scope'] != 'real_nuscenes_mini_resworld_functional' or metadata['can_fallback_used']:
        raise RuntimeError('Real mini data and real CAN messages are required.')
    for item in metadata['files']:
        if sha256(path / item['path']) != item['sha256']:
            raise RuntimeError('Real input checksum differs: ' + item['path'])
    if set(metadata['train_scenes']) & set(metadata['val_scenes']):
        raise RuntimeError('Training/validation scenes overlap.')
    return metadata


def create_model(metadata):
    from npu_compat import restore_trainable_resnet_pool
    cfg = config()
    from mmdet3d.models import build_model
    model = build_model(cfg.model)
    # Original initialization; the functional test does not use GeoBEV weights.
    model.init_weights()
    restore_trainable_resnet_pool(model.img_backbone)
    return model


def create_optimizer(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=0.01)


class RealFrames(Dataset):
    def __init__(self, dataset, validation):
        self.dataset, self.validation = dataset, validation
        self.indices = [i for i, info in enumerate(dataset.data_infos)
            if info['frame_idx'] >= 2 and info['fut_valid_flag']]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        result = self.dataset[original]
        if result is None:
            raise RuntimeError('Original real data pipeline returned None.')
        return {'sample': result, 'validation': self.validation,
            'token': self.dataset.data_infos[original]['token']}


def datasets(path, metadata):
    from mmdet3d.datasets import build_dataset
    cfg = config()
    result = []
    for split in ['train', 'val']:
        settings = cfg.data[split].copy()
        settings.pop('samples_per_gpu', None)
        settings.update(data_root=str(path), ann_file=str(path / ('vad_nuscenes_infos_temporal_' + split + '.pkl')),
            map_classes=cfg.map_classes, map_fixed_ptsnum_per_line=20, filter_empty_gt=False)
        if split == 'val':
            settings.update(test_mode=True, map_ann_file=str(path / 'nuscenes_map_anns_val.json'))
        result.append(RealFrames(build_dataset(settings), split == 'val'))
    return tuple(result)


def collate(examples):
    from mmcv.parallel import collate as original_collate, DataContainer
    raw = original_collate([item['sample'] for item in examples], samples_per_gpu=len(examples))
    validation = examples[0]['validation']

    def unwrap(value):
        if isinstance(value, DataContainer):
            return unwrap(value.data[0])
        if isinstance(value, (tuple, list)):
            return [unwrap(v) for v in value]
        return value

    # These are the actual tensors consumed by the original planning/depth losses.
    # Unused annotation objects are excluded from the safe checkpoint reference.
    batch = {}
    keys = ['img_inputs', 'can_bus', 'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat']
    if not validation:
        keys.append('gt_depth')
    for key in keys:
        value = raw[key]
        if validation:
            value = value[0]  # One official test augmentation.
        batch[key] = unwrap(value)
    batch['tokens'] = [item['token'] for item in examples]
    batch['img_metas'] = [{'sample_idx': item['token']} for item in examples]
    return batch


def transfer(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [transfer(item, device) for item in value]
    return value


def predict(model, batch, device):
    batch = transfer(batch, device)
    features, _, _ = model.extract_feat(None, img=batch['img_inputs'], img_metas=batch['img_metas'])
    outputs = model.pts_bbox_head([features[0], batch['can_bus']], batch['img_metas'],
        ego_his_trajs=batch['ego_his_trajs'], ego_lcf_feat=batch['ego_lcf_feat'], cmd=batch['ego_fut_cmd'])
    return outputs['ego_fut_preds']


def loss(model, batch, device):
    moved = transfer(batch, device)
    moved.pop('tokens')
    losses = model(return_loss=True, **moved)
    if set(losses) != {'loss_depth', 'loss_plan_reg', 'loss_plan_reg_init'}:
        raise RuntimeError('Original depth and both planning losses are required.')
    total = sum(value.mean() for value in losses.values())
    return total, {'real_tokens': batch['tokens'], 'real_can_data': True,
        'losses': {key: float(value.detach().cpu()) for key, value in losses.items()}}


def evaluate(model, dataset, device, batch_size, metadata):
    model.eval()
    totals = {1: 0.0, 2: 0.0, 3: 0.0}
    final = 0.0
    count = 0
    tokens = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate):
            prediction = predict(model, batch, device).cpu()
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError('Planning prediction contains nonfinite values.')
            command = batch['ego_fut_cmd'].reshape(-1, 3).argmax(-1)
            chosen = prediction[torch.arange(len(prediction)), command].cumsum(1)
            truth = batch['ego_fut_trajs'].reshape(-1, 6, 2).cumsum(1)
            masks = batch['ego_fut_masks'].reshape(-1, 6)
            if not bool((masks == 1).all()):
                raise RuntimeError('Evaluation frames must have all six valid future steps.')
            error = torch.linalg.vector_norm(chosen - truth, dim=-1)
            for seconds in totals:
                totals[seconds] += float(error[:, :seconds*2].mean(-1).sum())
            final += float(error[:, -1].sum())
            count += len(prediction)
            tokens += batch['tokens']
    return {'held_out_frames': count, 'validation_tokens': tokens,
        'l2_average_1s_m': totals[1]/count, 'l2_average_2s_m': totals[2]/count,
        'l2_average_3s_m': totals[3]/count, 'final_displacement_3s_m': final/count,
        'protocol': 'command-selected cumulative offsets, mean Euclidean error at 2/4/6 steps; all six futures valid',
        'navigation_command': 'official annotation command; not independently predicted',
        'collision_metrics_validated': False, 'random_initialization': True, 'formal_accuracy_claimed': False}
