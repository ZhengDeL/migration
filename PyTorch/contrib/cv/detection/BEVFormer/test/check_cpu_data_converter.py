"""Compare every converted mini annotation field with the independently delivered reference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pickle
from collections import Counter
import numpy as np
import argparse
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output_dir', type=Path, required=True)
args = parser.parse_args()
candidate = args.output_dir.resolve()
if not candidate.is_relative_to(root) or candidate.exists():
    raise RuntimeError('Use a fresh output directory inside this model')
if any(Path('/dev').glob('davinci[0-9]*')):
    raise RuntimeError('Run the CPU converter control without NPU device mounts')
sys.path.insert(0, str(root / 'tools'))
from prepare import require_checkout, patch_for, compat_patch_for, verify_applied_patch
source = root / 'sources/BEVFormer'
require_checkout('bevformer', source)
verify_applied_patch(source, patch_for('bevformer', root / 'vendor'), (compat_patch_for('bevformer'),))
from cpu_data_converter import load_author_converter
module, loading = load_author_converter(source / 'tools/data_converter/nuscenes_converter.py')
candidate.mkdir(parents=True)
module.create_nuscenes_infos(str((root / 'data/nuscenes').resolve()), str(candidate),
    str(root / 'data'), 'nuscenes', version='v1.0-mini', max_sweeps=10)
reference = root / 'prepared/nuscenes'
counts = Counter()
differences = []


def path_value(value):
    text = value.replace('\\', '/')
    parts = text.split('/')
    for name in ('samples', 'sweeps'):
        if name in parts:
            return '/'.join(parts[parts.index(name):])
    return value


def compare(a, b, position):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            differences.append(dict(field=position, kind='keys', reference=sorted(a), candidate=sorted(b)))
            return
        for key in a:
            compare(a[key], b[key], position + '.' + key)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            differences.append(dict(field=position, kind='length', reference=len(a), candidate=len(b)))
            return
        counts['sequences'] += 1
        for index, (left, right) in enumerate(zip(a, b)):
            compare(left, right, position + '[' + str(index) + ']')
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        counts['arrays'] += 1
        equal = a.shape == b.shape and a.dtype == b.dtype
        if equal:
            equal = np.array_equal(a, b, equal_nan=True) if np.issubdtype(a.dtype, np.floating) else np.array_equal(a, b)
        if not equal:
            differences.append(dict(field=position, kind='array', reference_shape=list(a.shape), candidate_shape=list(b.shape),
                                    reference_dtype=str(a.dtype), candidate_dtype=str(b.dtype)))
    else:
        counts['scalars'] += 1
        if isinstance(a, str) and isinstance(b, str):
            equal = path_value(a) == path_value(b)
        elif isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b):
            equal = True
        else:
            equal = a == b
        if not equal:
            differences.append(dict(field=position, kind='value', reference=repr(a)[:200], candidate=repr(b)[:200]))


files = []
for split, expected in [('train', 323), ('val', 81)]:
    name = 'nuscenes_infos_temporal_' + split + '.pkl'
    with (reference / name).open('rb') as stream:
        original = pickle.load(stream)
    with (candidate / name).open('rb') as stream:
        converted = pickle.load(stream)
    assert len(original['infos']) == len(converted['infos']) == expected
    compare(original, converted, split)
    files.append(dict(split=split, frames=expected,
        reference_sha256=hashlib.sha256((reference / name).read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((candidate / name).read_bytes()).hexdigest()))
report = dict(result='pass_all_original_mini_annotation_fields' if not differences else 'fail_annotation_field_difference',
    collected_utc=datetime.now(timezone.utc).isoformat(), counts=dict(counts),
    difference_count=len(differences), first_differences=differences[:30], files=files,
    path_comparison='Only the samples/sweeps directory prefix is rebased; numeric arrays, class labels and metadata compared',
    npu_devices_exposed=False, model_training_executed=False, full_trainval_conversion_completed=False, converter_loading=loading)
(candidate / 'annotation_field_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
assert not differences
