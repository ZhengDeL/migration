"""Check all mini fields and execute original train-only 900-anchor generation on CPU."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise RuntimeError('Use a fresh output directory inside this model')
    if any(Path('/dev').glob('davinci[0-9]*')):
        raise RuntimeError('Expose no NPU devices for this CPU control')
    sys.path.insert(0, str(ROOT / 'tools'))
    from prepare import require_checkout, patch_for, verify_applied_patch
    from cpu_data_converter import load_author_converter, load_author_anchor_generator
    source = ROOT / 'sources/Sparse4D'
    require_checkout('sparse4d', source)
    verify_applied_patch(source, patch_for('sparse4d', ROOT / 'vendor'))
    converter, loading = load_author_converter(source / 'tools/nuscenes_converter.py')
    output.mkdir(parents=True)
    converter.create_nuscenes_infos(str(ROOT / 'data/nuscenes'), str(output / 'nuscenes'),
                                    version='v1.0-mini', max_sweeps=10)
    counts, differences = Counter(), []

    def normalize(value):
        parts = value.replace('\\', '/').split('/')
        for key in ('samples', 'sweeps'):
            if key in parts:
                return '/'.join(parts[parts.index(key):])
        return value

    def compare(left, right, position):
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                differences.append(dict(field=position, kind='keys'))
                return
            for key in left:
                compare(left[key], right[key], position + '.' + key)
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            counts['sequences'] += 1
            if len(left) != len(right):
                differences.append(dict(field=position, kind='length'))
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f'{position}[{index}]')
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            counts['arrays'] += 1
            equal = left.shape == right.shape and left.dtype == right.dtype
            if equal:
                equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.floating) else np.array_equal(left, right)
            if not equal:
                differences.append(dict(field=position, kind='array'))
        else:
            counts['scalars'] += 1
            if isinstance(left, str) and isinstance(right, str):
                equal = normalize(left) == normalize(right)
            elif isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)) and np.isnan(left) and np.isnan(right):
                equal = True
            else:
                equal = left == right
            if not equal:
                differences.append(dict(field=position, kind='scalar', reference=repr(left)[:100], actual=repr(right)[:100]))

    records = []
    for split, expected in [('train', 323), ('val', 81)]:
        name = 'nuscenes_infos_' + split + '.pkl'
        reference = ROOT / 'prepared/nuscenes' / name
        generated = output / name
        with reference.open('rb') as stream:
            before = pickle.load(stream)
        with generated.open('rb') as stream:
            after = pickle.load(stream)
        if len(before['infos']) != expected or len(after['infos']) != expected:
            raise RuntimeError('Original mini split coverage differs')
        compare(before, after, split)
        records.append(dict(split=split, frames=expected,
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest()))
    report = dict(collected_utc=datetime.now(timezone.utc).isoformat(), model='Sparse4D',
        result='fail' if differences else 'pass_all_original_mini_annotation_fields',
        counts=dict(counts), difference_count=len(differences), first_differences=differences[:30],
        files=records, converter_loading=loading, npu_devices_exposed=False,
        full_trainval_conversion_completed=False, formal_accuracy_accepted=False)
    (output / 'cpu_preparation_control.json').write_text(json.dumps(report, indent=2) + '\n')
    if differences:
        raise RuntimeError('Original mini annotation fields differ; inspect the preserved report')
    anchors, anchor_loading = load_author_anchor_generator(source / 'tools/anchor_generator.py',
        source / 'projects/mmdet3d_plugin/core/box3d.py')
    np.random.seed(42)
    anchor_path = output / 'mini_train_only_kmeans900.npy'
    anchors.get_kmeans_anchor(str(output / 'nuscenes_infos_train.pkl'), num_anchor=900,
                             detection_range=55, output_file_name=str(anchor_path))
    value = np.load(anchor_path, allow_pickle=False)
    if value.shape != (900, 11) or not np.isfinite(value).all():
        raise RuntimeError('Original CPU anchor generator returned invalid anchors')
    report.update(result='pass_original_mini_conversion_and_train_only_anchor_generation',
        anchor_generation=dict(loading=anchor_loading, seed=42, num_anchor=900,
            detection_range=55, input_split='mini_train_only', input_frames=323,
            shape=list(value.shape), sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest()))
    (output / 'cpu_preparation_control.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
