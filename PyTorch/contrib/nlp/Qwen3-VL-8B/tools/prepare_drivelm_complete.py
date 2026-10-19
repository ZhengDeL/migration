"""Prepare the complete publisher DriveLM images and a deterministic scene-disjoint local evaluation split."""

import collections, concurrent.futures, hashlib, json, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/drivelm'
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
ASSETS = {
    'drivelm_nus_imgs_train.zip': (
        3483205396,
        '2cb95fafac00ca058c04735c339a183657d60962afee12d782b410c602eab936',
        24432,
    ),
    'drivelm_nus_imgs_val.zip': (
        704864335,
        'ded1b140a2c6f1b551d92555008c39a3c0ec3fd0825095e7aa920a5bbe6592e9',
        4794,
    ),
}
RAW_SHA = 'ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd'
EMPTY_ID = ('f55c8c62235146d895f611176fd0a5dc', 'ff0d87987502472982ed3647c81236d9', 'planning', 18)
SEED = 'drivelm-nus-v1.1-scene-holdout-ascend-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, value):
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def media_path(raw):
    parts = Path(raw).parts
    if 'samples' in parts:
        parts = parts[parts.index('samples') + 1 :]
    elif 'val_data' in parts:
        parts = parts[parts.index('val_data') + 1 :]
    else:
        raise RuntimeError('Unexpected official image path: ' + raw)
    assert len(parts) == 2 and parts[0] in CAMERAS and parts[1].lower().endswith('.jpg')
    p = (DATA / 'nuscenes/samples' / Path(*parts)).resolve()
    assert p.is_relative_to((DATA / 'nuscenes/samples').resolve())
    return p


def row(scene, frame, kind, index, entry, paths):
    # Exactly six camera placeholders, in a fixed named order. No future labels are added to prompts.
    prefix = '\n'.join(name + ': <image>' for name in CAMERAS) + '\n'
    return {
        'id': ':'.join((scene, frame, kind, str(index))),
        'scene_id': scene,
        'frame_id': frame,
        'task': kind,
        'messages': [
            {'role': 'user', 'content': prefix + entry['Q']},
            {'role': 'assistant', 'content': entry['A']},
        ],
        'images': [str(p) for p in paths],
    }


def iter_rows(raw, dev_scenes):
    for scene, document in raw.items():
        for frame, details in document['key_frames'].items():
            assert set(details['image_paths']) == set(CAMERAS)
            paths = [media_path(details['image_paths'][name]) for name in CAMERAS]
            for kind, entries in details['QA'].items():
                for index, entry in enumerate(entries):
                    if (scene, frame, kind, index) == EMPTY_ID:
                        assert isinstance(entry['Q'], str) and entry['Q'] and entry['A'] == ''
                        continue
                    assert (
                        isinstance(entry['Q'], str)
                        and entry['Q'].strip()
                        and isinstance(entry['A'], str)
                        and entry['A'].strip()
                    )
                    yield 'dev' if scene in dev_scenes else 'train', row(
                        scene, frame, kind, index, entry, paths
                    )


def main():
    from PIL import Image

    start = time.monotonic()
    DATA.mkdir(parents=True, exist_ok=True)
    status = {'state': 'verifying_complete_official_sources'}
    status_path = DATA / 'preparation_status.json'
    record(status_path, status)
    raw_path = DATA / 'raw/v1_1_train_nus.json'
    assert raw_path.stat().st_size == 192961829 and sha(raw_path) == RAW_SHA
    receipts = []
    image_sets = {}
    for name, (size, digest, expected_images) in ASSETS.items():
        path = DATA / 'archives' / name
        assert path.stat().st_size == size and sha(path) == digest
        status.update(state='extracting_publisher_images', asset=name)
        record(status_path, status)
        with zipfile.ZipFile(path) as archive:
            names = set()
            for item in archive.infolist():
                if item.is_dir():
                    continue
                assert item.filename.lower().endswith('.jpg')
                assert ((item.external_attr >> 16) & 0o170000) != 0o120000
                target = media_path(item.filename)
                assert target not in names
                names.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                # Read verifies each ZIP entry CRC; never accept a leftover partial extraction.
                with archive.open(item) as source, target.open('wb') as output:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block)
            assert len(names) == expected_images
            image_sets[name] = names
            receipts.append({'file': name, 'bytes': size, 'sha256': digest, 'images': len(names)})
    assert not (image_sets['drivelm_nus_imgs_train.zip'] & image_sets['drivelm_nus_imgs_val.zip'])
    status.update(state='decoding_and_hashing_every_image')
    record(status_path, status)

    def check_image(path):
        with Image.open(path) as image:
            image.load()
            dimensions = list(image.size)
            assert image.format == 'JPEG'
        return {
            'path': path.relative_to(DATA).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': sha(path),
            'dimensions': dimensions,
        }

    all_images = sorted(set.union(*image_sets.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        media = list(pool.map(check_image, all_images))
    raw = json.loads(raw_path.read_text())
    assert len(raw) == 696
    scenes = sorted(raw, key=lambda scene: hashlib.sha256((SEED + scene).encode()).hexdigest())
    dev = set(scenes[:70])
    train = set(scenes[70:])
    assert len(train) == 626 and not train & dev
    prepared = DATA / 'prepared'
    prepared.mkdir(exist_ok=True)
    counts = collections.Counter()
    tasks = {'train': collections.Counter(), 'dev': collections.Counter()}
    used = set()
    streams = {
        split: (prepared / (split + '.jsonl')).open('w', encoding='utf8', newline='\n')
        for split in ['train', 'dev']
    }
    try:
        for split, item in iter_rows(raw, dev):
            assert item['id'] not in used
            used.add(item['id'])
            counts[split] += 1
            tasks[split][item['task']] += 1
            for name in item['images']:
                assert Path(name) in image_sets['drivelm_nus_imgs_train.zip']
            streams[split].write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
    finally:
        for stream in streams.values():
            stream.close()
    assert sum(counts.values()) == 377955
    registry = {}
    for split in ['train', 'dev']:
        registry['drivelm_nus_v11_' + split] = {
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
    record(prepared / 'dataset_info.json', registry)
    excluded = {
        'id': list(EMPTY_ID),
        'reason': 'The sole publisher record with an empty answer; no fabricated label',
        'raw_qa_pairs': 377956,
        'usable_qa_pairs': 377955,
    }
    record(prepared / 'excluded_records.json', excluded)
    metadata = {
        'schema_version': 1,
        'dataset': 'DriveLM-nuScenes v1.1',
        'publisher_revision': 'ae973fbd4e8d4684af4ab234d504bd6c5e946868',
        'result': 'pass_complete_source_media_decode_and_scene_split_pending_context_audit',
        'raw_annotation_sha256': RAW_SHA,
        'scenes': 696,
        'frames': 4072,
        'raw_qa_pairs': 377956,
        'usable_qa_pairs': 377955,
        'excluded_records': excluded,
        'split_protocol': 'Local deterministic 626/70 scene split of official public training data; not the official hidden val score',
        'split_seed': SEED,
        'train_scenes': sorted(train),
        'dev_scenes': sorted(dev),
        'qa_counts': dict(counts),
        'qa_by_task': {k: dict(v) for k, v in tasks.items()},
        'camera_order': list(CAMERAS),
        'archives': receipts,
        'images_verified': len(media),
        'train_images': 24432,
        'official_val_images': 4794,
        'all_images_fully_decoded': True,
        'prepared_sha256': {
            name: sha(prepared / name)
            for name in ['train.jsonl', 'dev.jsonl', 'dataset_info.json', 'excluded_records.json']
        },
        'media_manifest_sha256': None,
        'seconds': time.monotonic() - start,
        'complete_npu_training_executed': False,
    }
    record(DATA / 'media_manifest.json', media)
    metadata['media_manifest_sha256'] = sha(DATA / 'media_manifest.json')
    record(DATA / 'metadata.json', metadata)
    status.update(
        state=metadata['result'], seconds=time.monotonic() - start, qa_counts=dict(counts), images=len(media)
    )
    record(status_path, status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
