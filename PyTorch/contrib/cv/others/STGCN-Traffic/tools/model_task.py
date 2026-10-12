"""Traffic STGCN from the pinned PyTorch implementation; no skeleton network."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate
from common import ROOT, sha256

sys.path.insert(0, str(ROOT / 'vendor'))


def check_data(path, mode):
    metadata = json.loads((path / 'metadata.json').read_text())
    if mode == 'full' and metadata['scope'] != 'pemsd7-m_full_chronological':
        raise RuntimeError('完整训练入口需要完整 PeMSD7(M) 时间划分；功能子集不满足该入口。')
    for item in metadata['files']:
        if sha256(path / item['path']) != item['sha256']:
            raise RuntimeError('交通数据 / 图矩阵 SHA256 不一致：' + item['path'])
    return metadata


def create_model(metadata):
    from model.models import STGCNChebGraphConv
    gso = np.load(ROOT / metadata['gso_path'], allow_pickle=False)
    args = SimpleNamespace(Kt=3, Ks=3, n_his=12, act_func='glu', graph_conv_type='cheb_graph_conv',
        gso=torch.from_numpy(gso), enable_bias=True, droprate=0.5)
    return STGCNChebGraphConv(args, [[1], [64,16,64], [64,16,64], [128,128], [1]], metadata['nodes'])


def create_optimizer(model, settings):
    return torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=0.001)


def datasets(path, metadata):
    result = []
    for split in ['train', 'val']:
        with np.load(path / (split + '.npz'), allow_pickle=False) as file:
            result.append(TensorDataset(torch.from_numpy(file['x'].copy()), torch.from_numpy(file['y'].copy())))
    return tuple(result)


collate = default_collate


def predict(model, batch, device):
    return model(batch[0].to(device)).reshape(len(batch[0]), -1)


def loss(model, batch, device):
    return torch.nn.functional.mse_loss(predict(model, batch, device), batch[1].to(device)), {'traffic_windows':len(batch[0])}


def evaluate(model, dataset, device, batch_size, metadata):
    model.eval()
    mean = np.asarray(metadata['mean'], dtype=np.float32)
    std = np.asarray(metadata['std'], dtype=np.float32)
    absolute = squared = percentage = 0.0
    points = nonzero = windows = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            prediction = predict(model, batch, device).cpu().numpy() * std + mean
            truth = batch[1].numpy() * std + mean
            if not np.isfinite(prediction).all():
                raise RuntimeError('独立验证交通预测含非有限值。')
            error = prediction - truth
            mask = np.abs(truth) > 1e-6
            absolute += float(np.abs(error).sum())
            squared += float(np.square(error).sum())
            percentage += float(np.abs(error[mask] / truth[mask]).sum())
            points += truth.size
            nonzero += int(mask.sum())
            windows += len(truth)
    return {'held_out_windows':windows, 'mae':absolute/points, 'rmse':(squared/points)**0.5,
        'mape_percent':100*percentage/nonzero, 'forecast_minutes':15,
        'metric_units':'original_speed_values', 'formal_accuracy_claimed':False}
