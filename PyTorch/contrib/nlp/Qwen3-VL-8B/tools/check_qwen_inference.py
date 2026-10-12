#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Check saved Qwen LoRA inference using one independent synthetic image."""
import argparse
import json
import time
from pathlib import Path
from fetch_qwen import verify, sha256_file
from check_qwen_checkpoint import check

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--adapter-dir', required=True, type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve(strict=True)
    adapter_dir = args.adapter_dir.resolve(strict=True)
    verify(model_dir)
    import torch
    import torch_npu
    from peft import PeftModel
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    artifacts = check(adapter_dir, 10, 2)
    fixture = Image.new('RGB', (224, 224), 'white')
    ImageDraw.Draw(fixture).rectangle((50, 50, 173, 173), fill=(255, 0, 0))
    image_path = adapter_dir / 'heldout_red_square.png'
    fixture.save(image_path)
    question = 'Describe the geometric shape and its color in this image.'
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)}, {'type': 'text', 'text': question}]}]
    report = {'model': 'qwen3vl8b', 'test': 'saved_adapter_image_inference', 'adapter_sha256': artifacts['adapter_sha256'], 'fixture_sha256': sha256_file(image_path), 'question': question, 'physical_cards': 1, 'precision': 'BF16', 'result': 'fail', 'scope': 'one independent synthetic image; no business accuracy or performance claim'}
    started = time.monotonic()
    try:
        torch.npu.set_device(0)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False).to('npu:0').eval()
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[fixture], padding=True, return_tensors='pt').to('npu:0')
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        torch.npu.synchronize()
        answer = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        report['answer'] = answer
        answer_lower = answer.lower()
        if 'red' not in answer_lower or not ('square' in answer_lower or 'rectangle' in answer_lower):
            raise RuntimeError('Saved adapter did not identify the shape and its color')
        report['result'] = 'pass_functional'
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds_including_model_load'] = round(time.monotonic() - started, 3)
        (adapter_dir / 'inference_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
if __name__ == '__main__':
    main()
