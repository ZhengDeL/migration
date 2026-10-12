"""Official basic Argoverse LSTM trajectory network on native NPU tensors."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, default_collate
from common import ROOT, sha256
from upstream_components import neural_classes, coordinate_functions


def check_data(path, mode):
    metadata = json.loads((path / 'metadata.json').read_text())
    if mode == 'full':
        raise RuntimeError('完整数据和上游 1/10/30 步 rollout 协议尚未验收；当前入口仅做真实轨迹功能验证。')
    if metadata['scope'] != 'official_argoverse_v11_train_val_functional_128_32' or len(metadata['train_sequences']) != 128 or len(metadata['val_sequences']) != 32:
        raise RuntimeError('The delivered official 128 train / 32 val trajectory subset is required.')
    for item in metadata['files']:
        if sha256(path / item['path']) != item['sha256']:
            raise RuntimeError('Real trajectory data SHA256 mismatch: ' + item['path'])
    if set(metadata['train_sequences']) & set(metadata['val_sequences']):
        raise RuntimeError('Trajectory train/validation sequence overlap.')
    if metadata['synthetic_data'] or metadata['coordinate_roundtrip_max_error_m'] > 1e-7:
        raise RuntimeError('Real coordinates and the original normalization round trip must pass.')
    return metadata


class MotionLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        Encoder, Decoder = neural_classes()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, history):
        real_batch = len(history)
        # The fixed runtime's native LSTMCell is unstable at NPU batch=1.
        # An independent duplicate row pads the kernel batch; it is excluded
        # from predictions, loss and metrics. The original cells are retained.
        if history.device.type == 'npu' and real_batch == 1:
            history = history.repeat(2, 1, 1)
        batch = len(history)
        state = (history.new_zeros((batch, 16)), history.new_zeros((batch, 16)))
        for index in range(history.shape[1]):
            state = self.encoder(history[:, index], state)
        current = history[:, -1]
        output = []
        for _ in range(30):
            current, state = self.decoder(current, state)
            output.append(current)
        return torch.stack(output, dim=1)[:real_batch]


def create_model(metadata):
    return MotionLSTM()


def create_optimizer(model, settings):
    # Independent per-parameter Adam state is retained for both original networks.
    return torch.optim.Adam(model.parameters(), lr=settings['lr'])


def datasets(path, metadata):
    result = []
    for split in ['train', 'val']:
        with np.load(path / (split + '.npz'), allow_pickle=False) as data:
            dataset = TensorDataset(torch.from_numpy(data['x'].copy()), torch.from_numpy(data['y'].copy()),
                                    torch.arange(len(data['x'])))
            dataset.coordinates = {key:data[key].copy() for key in ['truth_city', 'reference', 'translation', 'rotation']}
            result.append(dataset)
    return tuple(result)


collate = default_collate


def predict(model, batch, device):
    return model(batch[0].to(device))


def loss(model, batch, device):
    return torch.nn.functional.mse_loss(predict(model, batch, device), batch[1].to(device)), {
        'real_trajectory_sequences':len(batch[0]), 'rollout_steps':30,
        'sequence_indices':batch[2].tolist()}


def evaluate(model, dataset, device, batch_size, metadata):
    model.eval()
    functions = coordinate_functions()
    args = SimpleNamespace(use_delta=True, normalize=True, use_map=False,
                           obs_len=20, pred_len=30)
    ade = fde = misses = 0.0
    sequences = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            relative = predict(model, batch, device).cpu().numpy().astype(np.float64)
            indices = batch[2].numpy()
            helpers = {'REFERENCE':dataset.coordinates['reference'][indices],
                       'TRANSLATION':dataset.coordinates['translation'][indices],
                       'ROTATION':dataset.coordinates['rotation'][indices]}
            _, absolute = functions['get_abs_traj'](batch[0].numpy().astype(np.float64),
                                                   relative, args, helpers)
            if not np.isfinite(absolute).all():
                raise RuntimeError('Recovered trajectory contains nonfinite coordinates.')
            error = np.linalg.norm(absolute - dataset.coordinates['truth_city'][indices], axis=2)
            ade += float(error.mean(axis=1).sum())
            fde += float(error[:, -1].sum())
            misses += int((error[:, -1] > 2.0).sum())
            sequences += len(error)
    return {'held_out_sequences':sequences,'ade_m':ade/sequences,'fde_m':fde/sequences,
            'miss_rate':misses/sequences,'miss_threshold_m':2.0,'prediction_guesses_k':1,
            'observation_steps':20,'prediction_steps':30,'synthetic_data':False,
            'data_scope':metadata['scope'],'formal_accuracy_claimed':False}
