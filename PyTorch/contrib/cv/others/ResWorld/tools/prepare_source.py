"""Materialize pinned upstream source and explicit compatibility edits locally."""
import json
import shutil
from pathlib import Path
from common import ROOT, sha256, write_json


def prepare_source():
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    source = ROOT / 'vendor/ResWorld'
    target = ROOT / 'runtime/ResWorld'
    changes = []
    for item in lock['files']:
        original = ROOT / item['path']
        if sha256(original) != item['sha256']:
            raise RuntimeError('Upstream source differs: ' + item['path'])
        destination = target / item['upstream_path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = original.read_bytes()
        if item['upstream_path'].endswith('core/bbox/structures/lidar_box3d.py'):
            old = b'from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu'
            if content.count(old) != 1:
                raise RuntimeError('Unexpected upstream ROI import')
            content = content.replace(old, b'from mmcv.ops import points_in_boxes_part as points_in_boxes_gpu')
        elif item['upstream_path'].endswith('resworld/bevdet.py'):
            old = (b'        global2keyego = torch.inverse(keyego2global.double())\n'
                   b'        sensor2keyegos = \\\n'
                   b'            global2keyego @ ego2globals.double() @ sensor2egos.double()\n'
                   b'        sensor2keyegos = sensor2keyegos.float()')
            new = (b'        # Preserve upstream FP64 calibration calculations on CPU.\n'
                   b'        if keyego2global.device.type == "npu":\n'
                   b'            assert not any(t.requires_grad for t in (keyego2global, ego2globals, sensor2egos))\n'
                   b'            global2keyego = torch.inverse(keyego2global.cpu().double())\n'
                   b'            sensor2keyegos = (global2keyego @ ego2globals.cpu().double()\n'
                   b'                             @ sensor2egos.cpu().double()).float().to(ego2globals.device)\n'
                   b'        else:\n'
                   b'            global2keyego = torch.inverse(keyego2global.double())\n'
                   b'            sensor2keyegos = (global2keyego @ ego2globals.double()\n'
                   b'                             @ sensor2egos.double()).float()')
            if content.count(old) != 2:
                raise RuntimeError('Unexpected upstream calibration code')
            content = content.replace(old, new)
        elif item['upstream_path'] == 'tools/generate_point_label.py':
            content = content.replace(b'def generate(dataroot, save_dir):',
                b"def generate(dataroot, save_dir, version='v1.0-trainval'):")
            content = content.replace(b"NuScenes(version='v1.0-trainval', dataroot=dataroot", b'NuScenes(version=version, dataroot=dataroot')
        if not destination.exists() or destination.read_bytes() != content:
            part = destination.with_suffix(destination.suffix + '.partial')
            part.write_bytes(content)
            part.replace(destination)
        if content != original.read_bytes():
            changes.append({'path': item['upstream_path'], 'upstream_sha256': item['sha256'],
                'adapted_sha256': sha256(destination)})
    write_json(ROOT / 'runtime/patch_manifest.json', {'revision': lock['revision'], 'changes': changes})
    return target


if __name__ == '__main__':
    print(prepare_source())
