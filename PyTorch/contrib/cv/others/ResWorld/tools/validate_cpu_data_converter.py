"""Compare CPU-only author conversion with the existing original mini artifacts."""
import argparse
import json
import os
from pathlib import Path
from common import ROOT, own_path, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_path', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    a = p.parse_args()
    data, output = own_path(a.data_path, exists=True), own_path(a.output_dir)
    if output.exists():
        raise RuntimeError('Preserve earlier controls; select a new output directory.')
    metadata = json.loads((data / 'metadata.json').read_text())
    if (metadata['scope'] != 'real_nuscenes_mini_resworld_functional'
            or (metadata['train_frames'], metadata['val_frames']) != (323, 81)):
        raise RuntimeError('This component control requires the existing real mini baseline.')
    import glob
    if glob.glob('/dev/davinci[0-9]*'):
        raise RuntimeError('The CPU conversion control must not expose NPU devices.')
    import mmcv
    import numpy as np
    from cpu_data_converter import load_author_converter
    from prepare_source import prepare_source
    from prepare_data import load_module
    output.mkdir(parents=True)
    raw = output / 'mini_raw'
    for name in ['samples', 'v1.0-mini', 'maps', 'lidarseg/v1.0-mini', 'can_bus']:
        source, target = data / name, raw / name
        for base, dirs, files in os.walk(source):
            destination = target / Path(base).relative_to(source)
            destination.mkdir(parents=True, exist_ok=True)
            for filename in files:
                original = own_path(Path(base) / filename, exists=True)
                if not original.is_relative_to(data):
                    raise RuntimeError('The original mini file escapes the model dataset.')
                os.link(original, destination / filename)
    source = prepare_source()
    converter, loading = load_author_converter(source / 'tools/data_converter/vad_nuscenes_converter.py')
    converter.create_nuscenes_infos(str(raw), str(raw), str(raw), 'vad_nuscenes',
                                    version='v1.0-mini', max_sweeps=0)
    failures, arrays = [], 0

    def canonical(value, root):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in ['lidar_path', 'data_path'] and isinstance(item, str):
                    path = Path(item)
                    path = own_path(path if path.is_absolute() else ROOT / path, exists=True)
                    result[key] = path.relative_to(root).as_posix()
                else:
                    result[key] = canonical(item, root)
            return result
        if isinstance(value, (tuple, list)):
            return type(value)(canonical(item, root) for item in value)
        return value

    def compare(x, y, key):
        nonlocal arrays
        if isinstance(x, np.ndarray):
            arrays += 1
            equal = (isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape
                     and np.array_equal(x, y, equal_nan=x.dtype.kind in 'fc'))
            if not equal:
                failures.append(key)
        elif isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                failures.append(key + ':keys')
            else:
                for name in x:
                    compare(x[name], y[name], key + '/' + str(name))
        elif isinstance(x, (tuple, list)):
            if type(x) != type(y) or len(x) != len(y):
                failures.append(key + ':length')
            else:
                for i, (one, two) in enumerate(zip(x, y)):
                    compare(one, two, key + '/' + str(i))
        elif x != y and not (isinstance(x, (float, np.floating)) and np.isnan(x) and np.isnan(y)):
            failures.append(key)

    for split in ['train', 'val']:
        name = 'vad_nuscenes_infos_temporal_' + split + '.pkl'
        compare(canonical(mmcv.load(str(data / name)), data),
                canonical(mmcv.load(str(raw / name)), raw), split)
    depth = load_module(source / 'tools/generate_point_label.py', 'cpu_author_depth_control')
    depth.generate(str(raw), 'samples_point_label', version='v1.0-mini')
    count = 0
    for path in sorted((data / 'samples_point_label').rglob('*.npy')):
        relative = path.relative_to(data)
        one, two = np.load(path, allow_pickle=False), np.load(raw / relative, allow_pickle=False)
        if one.dtype != two.dtype or not np.array_equal(one, two):
            failures.append(relative.as_posix())
        count += 1
    if count != 2424 or len(list((raw / 'samples_point_label').rglob('*.npy'))) != 2424:
        failures.append('depth_file_coverage')
    result = {'result': 'pass_cpu_original_mini_conversion' if not failures else 'fail_cpu_mini_conversion',
        'npu_devices_exposed': False, 'baseline_metadata_sha256': sha256(data / 'metadata.json'),
        'train_frames': 323, 'val_frames': 81, 'annotation_arrays_compared': arrays,
        'depth_files_compared': count, 'mismatch_count': len(failures), 'mismatches': failures[:100],
        'path_comparison': 'Normalize only model-local sensor root locations before comparison',
        'annotation_protocol': 'Original mini converter; max_sweeps=0; full trainval max_sweeps=10 not tested here',
        'converter_loading': loading, 'positive_full_data_preparation_executed': False,
        'full_training_completed': False, 'full_accuracy_validated': False}
    write_json(output / 'cpu_conversion_control.json', result)
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError('CPU conversion differs from existing original mini artifacts.')


if __name__ == '__main__':
    main()
