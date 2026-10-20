"""Run original CPU conversion, full-config checks and mini-as-full rejection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    if not output.is_relative_to(ROOT) or (output / 'full_preparation_control.json').exists():
        raise RuntimeError('Use a fresh model-local output directory')
    if any(Path('/dev').glob('davinci[0-9]*')):
        raise RuntimeError('This control must expose no NPU devices')
    report = dict(collected_utc=datetime.now(timezone.utc).isoformat(), model='Sparse4D',
        result='fail', full_trainval_preparation_completed=False, full_training_completed=False,
        formal_accuracy_accepted=False, npu_devices_exposed=False)
    try:
        command = [sys.executable, str(ROOT / 'test/check_cpu_data_converter.py'),
                   '--output_dir', str(output / 'original_mini_control')]
        subprocess.run(command, check=True)
        report['original_mini_control'] = json.loads((output / 'original_mini_control/cpu_preparation_control.json').read_text())
        sys.path.insert(0, str(ROOT / 'tools'))
        from run_train import sparse_config
        from mmcv import Config
        source = ROOT / 'sources/Sparse4D'
        original = Config.fromfile(str(source / 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py'),
                                   import_custom_modules=False)
        full = Config.fromfile(str(sparse_config(source, output / 'full_config', False)), import_custom_modules=False)
        smoke = Config.fromfile(str(sparse_config(source, output / 'smoke_config', True, 'v1.0-mini')),
                                import_custom_modules=False)
        assert (full.total_batch_size, full.num_gpus, full.batch_size) == (12, 2, 6)
        assert full.optimizer.lr == 1.5e-4 and full.optimizer.foreach is False
        assert full.num_epochs == original.num_epochs == 100
        assert full.runner.type == original.runner.type == 'IterBasedRunner'
        assert full.runner.max_iters == full.num_iters_per_epoch * full.num_epochs == 234400
        assert dict(full.lr_config) == dict(original.lr_config)
        assert dict(full.fp16) == dict(original.fp16)
        assert (smoke.total_batch_size, smoke.batch_size, smoke.optimizer.lr, smoke.runner.max_iters) == (2, 1, 2.5e-5, 10)
        # Only the already verified modern serialization of the same backbone weights differs.
        full.model.img_backbone.pretrained = original.model.img_backbone.pretrained
        assert full.model == original.model
        report['full_config_projection'] = dict(result='pass_original_model_schedule_and_batch_contract',
            world_size=2, batch_per_rank=6, total_batch=12, learning_rate=1.5e-4,
            epochs=100, updates_per_epoch=2344, planned_updates=234400,
            original_model_unchanged=True, original_lr_schedule_unchanged=True,
            fp16=dict(full.fp16), historical_smoke_batch_and_steps_preserved=True,
            full_training_actually_executed=False)

        def reference_state():
            paths = [ROOT / 'prepared/nuscenes/nuscenes_infos_train.pkl',
                     ROOT / 'prepared/nuscenes/nuscenes_infos_val.pkl', ROOT / 'test/output/latest_run.json']
            result = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths if p.exists()}
            for name in ('nuscenes', 'nuscenes_anno_pkls'):
                result['source_link_' + name] = str((source / 'data' / name).resolve())
            return result
        before = reference_state()
        rejected = subprocess.run([sys.executable, str(ROOT / 'tools/prepare_full_data.py'),
            '--data_path', str(ROOT / 'data/nuscenes')], capture_output=True, text=True)
        after = reference_state()
        assert rejected.returncode != 0 and before == after
        assert 'v1.0-trainval' in rejected.stderr
        report['mini_as_full_rejection'] = dict(exit_code=rejected.returncode,
            stderr=rejected.stderr[-2000:], protected_before=before, protected_after=after,
            prepared_annotations_source_links_and_result_pointer_unchanged=True)
        profile = json.loads((ROOT / 'model.json').read_text())
        direct_output = output / 'rejected_direct_full_training'
        rejected = subprocess.run([sys.executable, str(ROOT / 'tools/run_train.py'),
            'sparse4d', '--source', str(source), '--sdk', str(ROOT / profile['sdk_dir']),
            '--devices', '0,1', '--output-dir', str(direct_output), '--execute'],
            capture_output=True, text=True)
        assert rejected.returncode != 0 and 'Complete original full-data preparation' in rejected.stderr
        assert before == reference_state() and not direct_output.exists()
        report['direct_full_entry_rejection'] = dict(exit_code=rejected.returncode,
            stderr=rejected.stderr[-2000:], no_training_config_or_environment_check_started=True,
            protected_state_unchanged=True)
        report['result'] = 'pass_original_cpu_preparation_full_config_and_negative_guard'
    except Exception as error:
        report['failure'] = repr(error)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'full_preparation_control.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
