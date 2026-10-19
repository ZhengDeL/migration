"""Audit every real QA using the pinned LLaMA-Factory multimodal SFT encoder, without truncation."""

import collections, hashlib, json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sources/LLaMA-Factory/src'))
import torch
from llamafactory.hparams import ModelArguments, DataArguments
from llamafactory.model import load_tokenizer
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

DATA = ROOT / 'data/drivelm'
torch.set_num_threads(4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    t = path.with_suffix('.writing')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(path)


def main():
    start = time.monotonic()
    meta = json.loads((DATA / 'metadata.json').read_text())
    args = ModelArguments(
        model_name_or_path=str(ROOT / 'weights/qwen3-vl-8b-instruct'),
        trust_remote_code=True,
        image_max_pixels=262144,
        video_max_pixels=16384,
        flash_attn='disabled',
    )
    loaded = load_tokenizer(args)
    tokenizer = loaded['tokenizer']
    processor = loaded['processor']
    data_args = DataArguments(template='qwen3_vl_nothink', cutoff_len=1000000, packing=False)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoder = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
    )
    native_metadata = template.mm_plugin._get_mm_token_metadata
    cache = {'key': None, 'value': None, 'frames': 0}

    def metadata_cache(images, videos, audios, processor):
        assert not videos and not audios
        key = tuple(images)
        if cache['key'] != key:
            value = native_metadata(images, videos, audios, processor)
            assert value is not None and 'image_grid_thw' in value
            cache.update(key=key, value=value, frames=cache['frames'] + 1)
        return cache['value']

    template.mm_plugin._get_mm_token_metadata = metadata_cache
    counts = collections.Counter()
    hist = collections.Counter()
    longest = {'length': 0}
    minimum = None
    label_min = None
    total = 0
    image_token = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    first = True
    for split in ['train', 'dev']:
        p = DATA / 'prepared' / (split + '.jsonl')
        assert sha(p) == meta['prepared_sha256'][p.name]
        with p.open() as stream:
            for line in stream:
                row = json.loads(line)
                messages = row['messages']
                images = row['images']
                assert len(images) == 6 and messages[0]['content'].count('<image>') == 6
                kwargs = {
                    'prompt': [messages[0]],
                    'response': [messages[1]],
                    'system': '',
                    'tools': '',
                    'images': images,
                    'videos': [],
                    'audios': [],
                }
                ids, labels = encoder._encode_data_example(**kwargs)
                if first:
                    template.mm_plugin._get_mm_token_metadata = native_metadata
                    original_ids, original_labels = encoder._encode_data_example(**kwargs)
                    assert ids == original_ids and labels == original_labels
                    template.mm_plugin._get_mm_token_metadata = metadata_cache
                    first = False
                expected = int(
                    (
                        cache['value']['image_grid_thw'].prod(dim=1)
                        // processor.image_processor.merge_size**2
                    ).sum()
                )
                assert ids.count(image_token) == expected and len(ids) == len(labels)
                answer_tokens = sum(x != -100 for x in labels)
                assert answer_tokens > 0
                label_min = answer_tokens if label_min is None else min(label_min, answer_tokens)
                length = len(ids)
                minimum = length if minimum is None else min(minimum, length)
                if length > longest['length']:
                    longest = {'length': length, 'id': row['id'], 'answer_tokens': answer_tokens}
                counts[split] += 1
                total += 1
                hist[str((length // 256) * 256)] += 1
                if total % 2000 == 0:
                    save(
                        DATA / 'context_audit_progress.json',
                        {
                            'state': 'running',
                            'rows': total,
                            'expected': 377955,
                            'max_tokens': longest['length'],
                            'seconds': time.monotonic() - start,
                        },
                    )
    assert total == 377955 and dict(counts) == meta['qa_counts'] and cache['frames'] == 4072
    cutoff = next((n for n in [2048, 4096, 8192] if longest['length'] <= n), None)
    result = {
        'result': (
            'pass_all_rows_original_sft_encoder_without_truncation'
            if cutoff
            else 'failed_context_budget_exceeds_verified_candidates'
        ),
        'dataset': 'DriveLM-nuScenes v1.1',
        'rows_checked': total,
        'qa_counts': dict(counts),
        'distinct_frames': cache['frames'],
        'encoding': 'Pinned LLaMA-Factory SupervisedDatasetProcessor._encode_data_example with cutoff=1000000',
        'multimodal_metadata_cache': 'One frame only; calls original _get_mm_token_metadata; first encoded row compared byte-for-byte to uncached original',
        'template': 'qwen3_vl_nothink',
        'image_max_pixels': 262144,
        'min_tokens': minimum,
        'max_tokens': longest['length'],
        'longest_record': longest,
        'minimum_supervised_answer_tokens': label_min,
        'target_cutoff_len': cutoff,
        'length_histogram_bucket_256': dict(hist),
        'source_annotation_sha256': meta['raw_annotation_sha256'],
        'prepared_sha256': meta['prepared_sha256'],
        'media_manifest_sha256': meta['media_manifest_sha256'],
        'seconds': time.monotonic() - start,
        'complete_npu_training_executed': False,
    }
    save(DATA / 'context_audit.json', result)
    save(DATA / 'context_audit_progress.json', {
        'state': result['result'], 'rows': total, 'expected': 377955,
        'max_tokens': longest['length'], 'seconds': result['seconds'],
    })
    print(json.dumps(result), flush=True)
    assert cutoff is not None, 'Context exceeds measured verified limits; do not train with silent truncation'


if __name__ == '__main__':
    main()
