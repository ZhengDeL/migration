#!/usr/bin/env python3
"""Verify and convert the publisher's question-only validation set, without labels."""
import collections
import hashlib
import json
from pathlib import Path

from prepare_drivelm_complete import CAMERAS, DATA, media_path, record, sha

VAL_BYTES = 9860626
VAL_BLOB_SHA1 = '1d5f1748f1f324d112af4ce691b00f8d5042cdbb'
VAL_SHA256 = 'b4d4be9e8cf6e6db691973b2711c0b58a752e08aeb3f505f2f89bcaf40be9cd8'
EMPTY_QUESTION = ('e8099a6136804f3bb9b38ff94d98eb64', 'd1d38dddb8174acf9cb776ec6c941024', 'planning', 1)


def prepare():
    path = DATA / 'raw/v1_1_val_nus_q_only.json'
    content = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if (
        len(content) != VAL_BYTES
        or blob != VAL_BLOB_SHA1
        or hashlib.sha256(content).hexdigest() != VAL_SHA256
    ):
        raise RuntimeError('Official validation questions differ from the pinned publisher object')
    raw = json.loads(content)
    train = json.loads((DATA / 'raw/v1_1_train_nus.json').read_text())
    if set(raw) & set(train):
        raise RuntimeError('Publisher validation scenes overlap public training scenes')
    manifest = json.loads((DATA / 'media_manifest.json').read_text())
    verified_images = {item['path']: item for item in manifest}
    prefix = '\n'.join(name + ': <image>' for name in CAMERAS) + '\n'
    counts = collections.Counter()
    frames = set()
    images = set()
    empty_questions = []
    output = DATA / 'prepared/official_val_questions.jsonl'
    with output.open('w', encoding='utf8', newline='\n') as stream:
        for scene, document in raw.items():
            for frame, details in document['key_frames'].items():
                if set(details['image_paths']) != set(CAMERAS):
                    raise RuntimeError('Validation frame must contain all six cameras')
                paths = [media_path(details['image_paths'][name]) for name in CAMERAS]
                frames.add(frame)
                for item in paths:
                    relative = item.relative_to(DATA).as_posix()
                    expected = verified_images.get(relative)
                    if not expected or not item.is_file() or sha(item) != expected['sha256']:
                        raise RuntimeError('Validation image is missing or changed: ' + relative)
                    images.add(relative)
                for kind, entries in details['QA'].items():
                    for index, entry in enumerate(entries):
                        if entry.get('A') != '' or not isinstance(entry.get('Q'), str):
                            raise RuntimeError('Unexpected publisher question-only schema')
                        identity = (scene, frame, kind, index)
                        empty = not entry['Q'].strip()
                        if empty:
                            if identity != EMPTY_QUESTION or entry['Q'] != '':
                                raise RuntimeError('Unexpected empty validation question')
                            empty_questions.append(list(identity))
                        row = {
                            'id': ':'.join((scene, frame, kind, str(index))),
                            'scene_id': scene,
                            'frame_id': frame,
                            'task': kind,
                            'question_index': index,
                            'messages': [{'role': 'user', 'content': prefix + entry['Q']}],
                            'images': [str(p) for p in paths],
                            'publisher_empty_question': empty,
                        }
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                        counts[kind] += 1
    if (len(raw), len(frames), len(images), sum(counts.values()), len(empty_questions)) != (
        149,
        799,
        4794,
        15480,
        1,
    ):
        raise RuntimeError('Official validation coverage does not match the pinned release')
    result = {
        'result': 'pass_official_questions_and_all_six_camera_images',
        'annotation_bytes': len(content),
        'annotation_sha256': VAL_SHA256,
        'publisher_git_blob_sha1': blob,
        'scenes': len(raw),
        'frames': len(frames),
        'images': len(images),
        'questions': sum(counts.values()),
        'questions_by_task': dict(counts),
        'nonempty_questions': sum(counts.values()) - len(empty_questions),
        'publisher_empty_question_entries_preserved': empty_questions,
        'empty_question_policy': 'Preserve the publisher entry and flag it; do not invent a question or a score',
        'train_scene_overlap': 0,
        'prepared_sha256': sha(output),
        'official_answers_available': False,
        'official_score_computed': False,
        'inference_executed': False,
    }
    record(DATA / 'official_validation_data.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    prepare()
