"""Pinned Ascend ModelZoo MobileNetV2 with the original 1000-class head."""
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, default_collate
from common import ROOT, sha256

sys.path.insert(0, str(ROOT / 'vendor'))


def check_data(path, mode):
    metadata = json.loads((path / 'metadata.json').read_text())
    expected_scope = 'imagenet_2012_full' if mode == 'full' else 'public_imagenette_functional_subset'
    expected_counts = {'train': 1281167, 'val': 50000} if mode == 'full' else {'train': 100, 'val': 40}
    if metadata['scope'] != expected_scope or metadata.get('counts') != expected_counts:
        raise RuntimeError('数据范围和完整样本数不匹配；不能把功能子集改名为 ImageNet 全集。')
    if metadata.get('head_classes') != 1000 or not metadata.get('training_validation_image_hashes_disjoint'):
        raise RuntimeError('必须保留原 1000 类和训练 / 验证图像内容互斥。')
    entries = metadata['files']
    if len({item['path'] for item in entries}) != len(entries):
        raise RuntimeError('数据清单出现重复路径。')
    for split, count in expected_counts.items():
        images = [item for item in entries if item['path'].startswith(split + '/')]
        if len(images) != count:
            raise RuntimeError('数据清单未覆盖全部图像：' + split)
    if {item['sha256'] for item in entries if item['path'].startswith('train/')} & {item['sha256'] for item in entries if item['path'].startswith('val/')}:
        raise RuntimeError('训练和验证集出现相同图像内容。')
    for item in metadata['files']:
        if sha256(path / item['path']) != item['sha256']:
            raise RuntimeError('图片 / 类别映射 SHA256 不一致：' + item['path'])
    return metadata


def create_model(metadata):
    from mobilenet import mobilenet_v2
    return mobilenet_v2(pretrained=False, num_classes=1000, width_mult=1.0)


def create_optimizer(model, settings):
    return torch.optim.SGD(model.parameters(), lr=settings['lr'], momentum=0.9, weight_decay=4e-5)


def datasets(path, metadata):
    from torchvision import datasets as vision_data, transforms
    classes = json.loads((path / 'class_index.json').read_text())
    mapping = {value[0]: int(index) for index, value in classes.items()}
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pipelines = [transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])]
    result = []
    for split, transform in zip(['train', 'val'], pipelines):
        dataset = vision_data.ImageFolder(str(path / split), transform=transform)
        if any(name not in mapping for name in dataset.classes):
            raise RuntimeError('类别目录必须为官方 ImageNet WNID。')
        old = {index: mapping[name] for name, index in dataset.class_to_idx.items()}
        dataset.samples = [(name, old[label]) for name, label in dataset.samples]
        dataset.imgs = dataset.samples
        dataset.targets = [label for name, label in dataset.samples]
        result.append(dataset)
    return tuple(result)


collate = default_collate


def predict(model, batch, device):
    return model(batch[0].to(device))


def loss(model, batch, device):
    prediction = predict(model, batch, device)
    target = batch[1].to(device)
    return torch.nn.functional.cross_entropy(prediction, target), {'images': int(target.numel())}


def evaluate(model, dataset, device, batch_size, metadata):
    model.eval()
    total = correct = correct_top5 = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            output = predict(model, batch, device)
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('独立验证图像输出不是有限数。')
            total += len(batch[1])
            correct += int((output.argmax(dim=1).cpu() == batch[1]).sum())
            correct_top5 += int((output.topk(5,dim=1).indices.cpu() == batch[1][:,None]).any(dim=1).sum())
    return {'held_out_images': total, 'correct_top1': correct, 'top1': correct / total,
        'correct_top5':correct_top5,'top5':correct_top5/total,
        'class_head': 1000, 'claim': 'subset_functionality_only' if metadata['scope'] != 'imagenet_2012_full' else 'measured_on_supplied_dataset'}


def adjust_learning_rate(optimizer, global_step, steps_per_epoch, settings):
    # Same step-decay formula as the pinned ModelZoo training reference.
    interval = max(1, int(steps_per_epoch * 2.5))
    value = settings['lr'] * (0.98 ** (global_step // interval))
    for group in optimizer.param_groups:
        group['lr'] = value
