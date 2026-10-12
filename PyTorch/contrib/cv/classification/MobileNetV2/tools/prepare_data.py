"""Prepare deterministic real Imagenette inputs without changing class meanings."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path
from common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imagenet_path', type=Path, help='optional model-local full ImageNet directory with train/val WNID classes')
    args = parser.parse_args()
    target = ROOT / 'data' / ('imagenet' if args.imagenet_path else 'functional')
    if (target / 'metadata.json').is_file():
        from model_task import check_data
        check_data(target, 'full' if args.imagenet_path else 'functional')
        print('Existing prepared inputs verified:', target)
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('保留现有数据目录，拒绝覆盖：' + str(target))
    mapping = ROOT / 'data/raw/imagenet_class_index.json'
    assets = json.loads((ROOT / 'assets.lock.json').read_text())['files']
    for item in assets:
        if sha256(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('公开原始数据摘要不一致：' + item['path'])
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mapping, target / 'class_index.json')
    counts = {}
    if args.imagenet_path:
        source = args.imagenet_path.resolve(strict=True)
        if not source.is_relative_to(ROOT) or source == target:
            raise RuntimeError('请把完整数据放入本模型内的另一目录，例如 data/imagenet_original。')
        canonical = {v[0] for v in json.loads(mapping.read_text()).values()}
        for split in ['train', 'val']:
            if {p.name for p in (source / split).iterdir() if p.is_dir()} != canonical:
                raise RuntimeError('完整 ImageNet 的类别集合必须包含全部 1000 个官方 WNID。')
            shutil.copytree(source / split, target / split)
        scope = 'imagenet_2012_full'
    else:
        selected = {split: {} for split in ['train', 'val']}
        with tarfile.open(ROOT / 'data/raw/imagenette2-160.tgz', 'r:gz') as archive:
            members = sorted((m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith('.jpeg')), key=lambda m: m.name)
            for member in members:
                parts = Path(member.name).parts
                if len(parts) != 4 or parts[1] not in ['train', 'val']:
                    continue
                split, wnid = parts[1:3]
                limit = 10 if split == 'train' else 4
                count = selected[split].get(wnid, 0)
                if count >= limit:
                    continue
                destination = target / split / wnid / parts[3]
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as input_file, destination.open('xb') as output:
                    shutil.copyfileobj(input_file, output)
                selected[split][wnid] = count + 1
        if len(selected['train']) != 10 or len(selected['val']) != 10:
            raise RuntimeError('Imagenette 类别数量与官方子集不匹配。')
        scope = 'public_imagenette_functional_subset'
    files = []
    for split in ['train', 'val']:
        samples = sorted(p for p in (target / split).rglob('*') if p.is_file())
        counts[split] = len(samples)
        files += [{'path': p.relative_to(target).as_posix(), 'sha256': sha256(p)} for p in samples]
    if args.imagenet_path and counts != {'train': 1281167, 'val': 50000}:
        raise RuntimeError('完整 ImageNet-2012 数量应为 train=1281167 / val=50000。')
    files.append({'path': 'class_index.json', 'sha256': sha256(target / 'class_index.json')})
    # Hash equality would indicate training/validation content leakage.
    train_hashes = {i['sha256'] for i in files if i['path'].startswith('train/')}
    val_hashes = {i['sha256'] for i in files if i['path'].startswith('val/')}
    if train_hashes & val_hashes:
        raise RuntimeError('训练和验证集出现相同图片内容。')
    write_json(target / 'metadata.json', {'model': 'mobilenetv2', 'scope': scope, 'counts': counts,
        'head_classes': 1000, 'image_size': 224, 'class_mapping_source': 'TensorFlow canonical ImageNet class_index',
        'training_validation_image_hashes_disjoint': True, 'files': files})
    print(json.dumps({'data_path': str(target), 'scope': scope, 'counts': counts}, indent=2))


if __name__ == '__main__':
    main()
