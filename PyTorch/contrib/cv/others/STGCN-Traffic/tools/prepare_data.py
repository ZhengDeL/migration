"""Prepare real PeMSD7 data with chronological splits and train-only scaling."""
import argparse
import io
import json
import sys
import zipfile
import numpy as np
import scipy.sparse as sparse
from common import ROOT, sha256, write_json

sys.path.insert(0, str(ROOT / 'vendor'))


def windows(values):
    count = len(values) - 12 - 3  # Match pinned upstream data_transform.
    x = np.stack([values[i:i+12] for i in range(count)])[:,None]
    y = np.stack([values[i+12+3-1] for i in range(count)])
    return x.astype(np.float32), y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'data' / ('full' if args.full else 'functional')
    if (target / 'metadata.json').is_file():
        from model_task import check_data
        check_data(target, 'full' if args.full else 'functional')
        print('Existing prepared traffic inputs verified:', target)
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('保留现有交通数据目录，拒绝覆盖。')
    for entry in json.loads((ROOT / 'assets.lock.json').read_text())['files']:
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('原始交通数据来源摘要不一致：' + entry['path'])
    with zipfile.ZipFile(ROOT / 'data/raw/PeMSD7_Full.zip') as archive:
        with archive.open('PeMSD7_V_228.csv') as stream:
            velocity = np.loadtxt(stream, delimiter=',', dtype=np.float64 if args.full else np.float32)
    if velocity.shape != (12672,228) or not np.isfinite(velocity).all():
        raise RuntimeError('作者发布的 PeMSD7(M) shape 或数值不符。')
    # Official PyTorch recipe: 70% / 15% / 15% chronological split.
    val_length = int(len(velocity)*0.15)
    train_length = len(velocity)-2*val_length
    ranges = {'train':[0,train_length], 'val':[train_length,train_length+val_length], 'test':[train_length+val_length,len(velocity)]}
    if not args.full:
        ranges = {name:[start,min(end,start+(512 if name=='train' else 256))] for name,(start,end) in ranges.items()}
    train = velocity[slice(*ranges['train'])]
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-6] = 1
    target.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name,(start,end) in ranges.items():
        x,y = windows((velocity[start:end]-mean)/std)
        counts[name] = len(x)
        np.savez_compressed(target / (name+'.npz'), x=x, y=y)
    from graph_utils import calc_gso, calc_chebynet_gso
    adjacency = sparse.load_npz(ROOT / 'data/raw/adj.npz')
    np.random.seed(42)
    gso = calc_chebynet_gso(calc_gso(adjacency, 'sym_norm_lap')).toarray().astype(np.float32)
    if gso.shape != (228,228) or not np.isfinite(gso).all():
        raise RuntimeError('图移位矩阵无效。')
    np.save(target / 'gso.npy', gso)
    files = [{'path':p.name,'sha256':sha256(p)} for p in sorted(target.iterdir()) if p.is_file()]
    metadata = {'model':'stgcn-traffic', 'scope':'pemsd7-m_full_chronological' if args.full else 'pemsd7-m_functional_subset',
        'nodes':228,'history_steps':12,'forecast_steps':3,'interval_minutes':5,
        'raw_rows':len(velocity),'raw_time_ranges':ranges,'counts':counts,
        'scaler_fit_raw_time_range':ranges['train'],'mean':mean.tolist(),'std':std.tolist(),
        'gso_path':(target / 'gso.npy').relative_to(ROOT).as_posix(),
        'train_val_test_raw_intervals_disjoint':True,'files':files}
    write_json(target / 'metadata.json', metadata)
    print(json.dumps({'scope':metadata['scope'],'ranges':ranges,'counts':counts},indent=2))


if __name__ == '__main__':
    main()
