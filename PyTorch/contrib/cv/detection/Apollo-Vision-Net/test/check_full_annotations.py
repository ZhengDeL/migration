"""Read-only complete annotation/label audit; sensor files are deliberately separate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw_metadata', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    args = parser.parse_args()
    output, raw = args.output_dir.resolve(strict=True), args.raw_metadata.resolve(strict=True)
    if not output.is_relative_to(ROOT) or not raw.is_relative_to(ROOT):
        raise RuntimeError('Use metadata and output belonging to this model directory')
    if (output / 'full_annotation_control.json').exists() or any(Path('/dev').glob('davinci[0-9]*')):
        raise RuntimeError('Use a fresh output and expose no NPU devices')
    sys.path.insert(0, str(ROOT / 'tools'))
    from full_annotation_checks import audit, NAMES
    from world_mini_data import digest
    assets, occ = ROOT / 'data/apollo', ROOT / 'data/apollo/occ_gt_release_v1_0'
    paths = [assets / name for name in NAMES]
    before = {p.name: digest(p) for p in paths}
    report = dict(collected_utc=datetime.now(timezone.utc).isoformat(), result='fail',
        raw_media_coverage_checked=False, full_trainval_preparation_completed=False,
        full_training_completed=False, formal_accuracy_accepted=False, npu_devices_exposed=False)
    try:
        report['original_full_annotation_audit'] = audit(paths, raw, occ, check_media=False)
        # Use the real mini intersections with a renamed version; counts must still reject them.
        candidates = []
        for original, mini_name in zip(paths, ('apollo_mini_train.pkl', 'apollo_mini_val.pkl')):
            with (ROOT / 'prepared/apollo' / mini_name).open('rb') as stream:
                document = pickle.load(stream)
            document['metadata']['version'] = 'v1.0-trainval'
            candidate = output / original.name
            with candidate.open('wb') as stream:
                pickle.dump(document, stream, pickle.HIGHEST_PROTOCOL)
            candidates.append(candidate)
        try:
            audit(candidates, raw, occ, check_media=False)
        except RuntimeError as error:
            if 'sample/token membership mismatch' not in str(error):
                raise
            report['renamed_subset_rejection'] = dict(result='pass_rejected', error=str(error))
        else:
            raise RuntimeError('Renamed mini annotations were incorrectly accepted as full')
        direct_output = output / 'rejected_direct_full_training'
        latest = ROOT / 'test/output/latest_run.json'
        pointer_before = digest(latest) if latest.exists() else None
        rejected = subprocess.run([sys.executable, str(ROOT / 'tools/run_train.py'),
            'apollo_visionnet', '--source', str(ROOT / 'sources/Apollo-Vision-Net'),
            '--devices', '0,1,2,3', '--output-dir', str(direct_output), '--execute'],
            capture_output=True, text=True)
        if (rejected.returncode == 0 or 'Complete original full-data preparation' not in rejected.stderr
                or direct_output.exists() or pointer_before != (digest(latest) if latest.exists() else None)):
            raise RuntimeError('Direct full entry bypassed the original full preparation gate')
        report['direct_full_entry_rejection'] = dict(exit_code=rejected.returncode,
            stderr=rejected.stderr[-2000:], no_training_config_or_environment_check_started=True,
            existing_result_pointer_unchanged=True)
        after = {p.name: digest(p) for p in paths}
        if before != after:
            raise RuntimeError('Original full annotation files changed')
        report.update(result='pass_original_full_annotations_and_renamed_subset_guard',
            original_annotation_sha256_before=before, original_annotation_sha256_after=after,
            original_labels_and_annotation_payloads_not_modified=True)
    except Exception as error:
        report['failure'] = repr(error)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'full_annotation_control.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
