#!/usr/bin/env python3
"""Verify complete DriveLM source identity, media, scene split and encoded context."""
import argparse
import hashlib
import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

from prepare_drivelm_complete import ASSETS, CAMERAS, DATA, RAW_SHA, ROOT, SEED, iter_rows, media_path, sha

TRAIN_FILE = 'v1_1_train_nus.json'
TRAIN_BYTES = 192961829
TRAIN_SHA256 = RAW_SHA
TRAIN_DATASET = 'drivelm_nus_v11_train'
DEV_DATASET = 'drivelm_nus_v11_dev'
PUBLISHER_REVISION = 'ae973fbd4e8d4684af4ab234d504bd6c5e946868'
COUNTS = {'train': 340119, 'dev': 37836}


def require(condition, message):
    if not condition:
        raise RuntimeError('DriveLM 完整数据校验失败：' + message)


def verify_official_annotations(data):
    data = Path(data).resolve(strict=True)
    require(data.is_relative_to(ROOT.resolve()), '数据必须位于本模型目录内')
    path = data / 'raw' / TRAIN_FILE
    if not path.is_file():
        path = data / TRAIN_FILE
    require(path.is_file() and path.stat().st_size == TRAIN_BYTES, '缺少完整官方 DriveLM v1.1 训练问答')
    digest = sha(path)
    require(digest == RAW_SHA, '训练问答与固定官方版本 SHA256 不符')
    return {
        'dataset': 'DriveLM-nuScenes v1.1',
        'publisher_revision': PUBLISHER_REVISION,
        'annotation_bytes': TRAIN_BYTES,
        'annotation_sha256': digest,
        'result': 'pass_annotation_identity_only',
        'media_coverage_verified': False,
        'formal_training_verified': False,
    }


def verify_media():
    """Compare every prepared image to bytes from the hash-locked publisher ZIPs."""
    manifest = json.loads((DATA / 'media_manifest.json').read_text())
    require(len(manifest) == 29226, '完整媒体清单应有 29,226 张图像')
    by_path = {item['path']: item for item in manifest}
    require(len(by_path) == len(manifest), '媒体清单存在重复图像')
    seen = set()
    for name, (size, digest, count) in ASSETS.items():
        archive = DATA / 'archives' / name
        require(
            archive.is_file() and archive.stat().st_size == size and sha(archive) == digest,
            '官方图像归档缺失或身份不符：' + name,
        )
        found = 0
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                if item.is_dir():
                    continue
                path = media_path(item.filename)
                relative = path.relative_to(DATA).as_posix()
                expected = by_path.get(relative)
                require(expected is not None and relative not in seen, '归档与媒体清单不一致：' + relative)
                h = hashlib.sha256()
                with zipped.open(item) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                require(
                    expected['sha256'] == h.hexdigest() and expected['bytes'] == item.file_size,
                    '媒体清单不对应官方归档：' + relative,
                )
                require(
                    path.is_file() and path.stat().st_size == item.file_size and sha(path) == h.hexdigest(),
                    '已解压图像缺失或内容变化：' + relative,
                )
                seen.add(relative)
                found += 1
        require(found == count, '官方图像归档条目数不符：' + name)
    require(seen == set(by_path), '媒体清单存在非官方额外条目')


def require_full_protocol(data, dataset_id):
    # Check before image/device/output changes; direct launchers call this too.
    require(dataset_id == TRAIN_DATASET, '正式数据集名必须为 ' + TRAIN_DATASET)
    actual = Path(data).resolve(strict=True) if data is not None else DATA.resolve(strict=True)
    require(
        actual == DATA.resolve() and actual.is_relative_to(ROOT.resolve()),
        '完整数据应准备在本模型 data/drivelm 下',
    )
    verify_official_annotations(actual)
    metadata = json.loads((DATA / 'metadata.json').read_text())
    raw = json.loads((DATA / 'raw' / TRAIN_FILE).read_text())
    scenes = sorted(raw, key=lambda scene: hashlib.sha256((SEED + scene).encode()).hexdigest())
    dev_scenes = set(scenes[:70])
    require(
        len(scenes) == 696
        and metadata['train_scenes'] == sorted(set(scenes[70:]))
        and metadata['dev_scenes'] == sorted(dev_scenes),
        '场景留出划分与固定协议不符',
    )
    require(
        metadata['qa_counts'] == COUNTS and metadata['camera_order'] == list(CAMERAS),
        '问答数量或相机顺序不符',
    )
    require(
        metadata['raw_annotation_sha256'] == RAW_SHA and metadata['all_images_fully_decoded'] is True,
        '缺少原始标注身份或全部图像解码检查',
    )
    require(sha(DATA / 'media_manifest.json') == metadata['media_manifest_sha256'], '媒体清单已变化')
    verify_media()
    prepared = DATA / 'prepared'
    expected_names = {'train.jsonl', 'dev.jsonl', 'dataset_info.json', 'excluded_records.json'}
    require(set(metadata['prepared_sha256']) == expected_names, '缺少或包含未知数据产物身份')
    for name, digest in metadata['prepared_sha256'].items():
        require(sha(prepared / name) == digest, '准备后的数据文件已变化：' + name)
    counts = {'train': 0, 'dev': 0}
    with ExitStack() as stack:
        streams = {
            split: stack.enter_context((prepared / (split + '.jsonl')).open(encoding='utf8'))
            for split in counts
        }
        for split, expected in iter_rows(raw, dev_scenes):
            line = streams[split].readline()
            require(bool(line) and json.loads(line) == expected, '转换内容与官方问答不符：' + expected['id'])
            counts[split] += 1
        require(all(not stream.readline() for stream in streams.values()), '转换结果包含额外问答')
    require(counts == COUNTS, '有效问答应为 340,119 训练条目和 37,836 留出条目')
    registry = json.loads((prepared / 'dataset_info.json').read_text())
    for split, name in [('train', TRAIN_DATASET), ('dev', DEV_DATASET)]:
        expected = {
            'file_name': split + '.jsonl',
            'formatting': 'sharegpt',
            'columns': {'messages': 'messages', 'images': 'images'},
            'tags': {
                'role_tag': 'role',
                'content_tag': 'content',
                'user_tag': 'user',
                'assistant_tag': 'assistant',
            },
        }
        require(registry.get(name) == expected, 'LLaMA-Factory 注册内容与完整协议不符')
    audit = json.loads((DATA / 'context_audit.json').read_text())
    require(
        audit['result'] == 'pass_all_rows_original_sft_encoder_without_truncation',
        '全部样本上下文检查尚未通过',
    )
    require(
        audit['rows_checked'] == 377955 and audit['qa_counts'] == COUNTS and audit['distinct_frames'] == 4072,
        '上下文检查覆盖范围不完整',
    )
    require(
        audit['prepared_sha256'] == metadata['prepared_sha256']
        and audit['source_annotation_sha256'] == RAW_SHA
        and audit['media_manifest_sha256'] == metadata['media_manifest_sha256'],
        '上下文检查不对应当前数据',
    )
    require(
        audit['template'] == 'qwen3_vl_nothink'
        and audit['image_max_pixels'] == 262144
        and audit['minimum_supervised_answer_tokens'] > 0,
        '编码配置或监督标签不符',
    )
    cutoff = audit['target_cutoff_len']
    require(cutoff in (2048, 4096, 8192) and 0 < audit['max_tokens'] <= cutoff, '上下文长度超过已验证上限')
    return {
        'result': 'pass_complete_drivelm_data_and_context',
        'dataset': TRAIN_DATASET,
        'eval_dataset': DEV_DATASET,
        'dataset_dir': str(prepared),
        'cutoff_len': cutoff,
        'train_examples': COUNTS['train'],
        'eval_examples': COUNTS['dev'],
        'prepared_sha256': metadata['prepared_sha256'],
        'max_tokens': audit['max_tokens'],
        'official_hidden_validation_score': False,
        'formal_training_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, default=DATA)
    parser.add_argument('--dataset_id', default=TRAIN_DATASET)
    parser.add_argument('--verify-annotations-only', action='store_true')
    args = parser.parse_args()
    report = (
        verify_official_annotations(args.data_path)
        if args.verify_annotations_only
        else require_full_protocol(args.data_path, args.dataset_id)
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc))
