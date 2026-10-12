"""Check actual model-local mini inputs, create original annotations/depth if absent."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from common import ROOT, sha256, write_json
from prepare_source import prepare_source


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    target = ROOT / 'data/nuscenes'
    source = prepare_source()
    if (target / 'metadata.json').is_file():
        from model_task import check_data
        check_data(target, 'functional')
        print('Existing real ResWorld data and all input hashes verified.')
        return
    for folder in ['samples', 'v1.0-mini', 'maps/expansion', 'lidarseg/v1.0-mini', 'can_bus']:
        if not (target / folder).is_dir():
            raise RuntimeError('Missing original mini input: ' + folder + '. See this model README.')
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    scenes = json.loads((target / 'v1.0-mini/scene.json').read_text())
    bus = NuScenesCanBus(dataroot=str(target))
    for scene in scenes:
        for kind in ['pose', 'steeranglefeedback']:
            messages = bus.get_messages(scene['name'], kind)
            if not messages or 'utime' not in messages[0]:
                raise RuntimeError('Missing original CAN messages: ' + scene['name'] + '/' + kind)
    import mmcv
    sys.path.insert(0, str(source))
    annotations = [target / ('vad_nuscenes_infos_temporal_' + split + '.pkl') for split in ['train', 'val']]
    if not all(p.is_file() for p in annotations):
        module = load_module(source / 'tools/data_converter/vad_nuscenes_converter.py', 'official_converter')
        module.create_nuscenes_infos(str(target), str(target), str(target), 'vad_nuscenes', version='v1.0-mini', max_sweeps=0)
    if len(list((target / 'samples_point_label').rglob('*.npy'))) != 404*6:
        module = load_module(source / 'tools/generate_point_label.py', 'official_depth')
        module.generate(str(target), 'samples_point_label', version='v1.0-mini')
    subsets = {}
    for split, path in zip(['train', 'val'], annotations):
        original = mmcv.load(str(path))
        for info in original['infos']:
            if not np.any(np.asarray(info['can_bus'])[7:16]):
                raise RuntimeError('Original CAN converter used a fallback vector.')
            # Normalize input filenames for relocation of the independent folder.
            paths = [(info, 'lidar_path')]
            paths += [(camera, 'data_path') for camera in info['cams'].values()]
            for record, key in paths:
                candidate = Path(record[key])
                candidate = candidate if candidate.is_absolute() else ROOT / candidate
                candidate = candidate.resolve(strict=True)
                if not candidate.is_relative_to(target.resolve()):
                    raise RuntimeError('Input is outside this independent model dataset.')
                record[key] = candidate.relative_to(ROOT).as_posix()
        # Atomic replacement also avoids changing any earlier delivery's hardlink.
        temporary = path.with_suffix('.writing.pkl')
        mmcv.dump(original, str(temporary))
        temporary.replace(path)
        subsets[split] = original['infos']
    train_scenes = sorted({item['scene_token'] for item in subsets['train']})
    val_scenes = sorted({item['scene_token'] for item in subsets['val']})
    if set(train_scenes) & set(val_scenes):
        raise RuntimeError('Official mini scene splits overlap.')
    files = [{'path': path.relative_to(target).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        for path in sorted(target.rglob('*')) if path.is_file() and path.name != 'metadata.json']
    metadata = {'model': 'resworld', 'scope': 'real_nuscenes_mini_resworld_functional',
        'train_frames': len(subsets['train']), 'val_frames': len(subsets['val']),
        'train_scenes': train_scenes, 'val_scenes': val_scenes,
        'can_message_files': len(scenes)*2, 'can_fallback_used': False,
        'foreground_depth_files': 404*6, 'six_cameras': True,
        'temporal_frames': 3, 'random_initialization': True, 'synthetic_data': False,
        'formal_accuracy_claimed': False, 'files': files}
    write_json(target / 'metadata.json', metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
