#!/usr/bin/env python3
"""Verify official DriveLM source assets; full training awaits its data protocol."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = 'v1_1_train_nus.json'
TRAIN_BYTES = 192961829
TRAIN_SHA256 = 'ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd'
PUBLISHER_REVISION = 'ae973fbd4e8d4684af4ab234d504bd6c5e946868'

def verify_official_annotations(data):
    data = Path(data).resolve(strict=True)
    if not data.is_relative_to(ROOT.resolve()):
        raise RuntimeError('DriveLM 数据必须位于本模型目录内')
    path = data / TRAIN_FILE
    if not path.is_file() or path.stat().st_size != TRAIN_BYTES:
        raise RuntimeError('缺少完整官方 DriveLM v1.1 训练问答：' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != TRAIN_SHA256:
        raise RuntimeError('DriveLM 训练问答与固定官方版本 SHA256 不符')
    return {'dataset': 'DriveLM-nuScenes v1.1', 'publisher_revision': PUBLISHER_REVISION,
            'annotation_bytes': TRAIN_BYTES, 'annotation_sha256': digest.hexdigest(),
            'result': 'pass_annotation_identity_only',
            'media_coverage_verified': False, 'formal_training_verified': False}

def require_full_protocol(data, dataset_id):
    # Renaming a six-row demo must never create a full-data acceptance claim.
    raise RuntimeError(
        'DriveLM 正式训练入口尚未完成：需要完整官方问答与六相机图像、'
        '场景划分、LLaMA-Factory 格式转换及全部样本上下文长度校验。'
        '修改 dataset_id 或提供一个已有目录不能通过完整训练校验。'
        '当前仅支持已实测的 functional 入口；完整协议通过实机验收后再开放。')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--dataset_id', default='drivelm_nus_v11_train')
    parser.add_argument('--verify-annotations-only', action='store_true',
                        help='仅核验官方训练问答身份，不验证媒体或完整训练')
    args = parser.parse_args()
    if args.verify_annotations_only:
        print(json.dumps(verify_official_annotations(args.data_path), indent=2))
    else:
        require_full_protocol(args.data_path, args.dataset_id)

if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError) as exc:
        raise SystemExit(str(exc))
