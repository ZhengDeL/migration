"""Regression: a renamed demo must not start full training or mutate result state."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FullDataGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='full_guard_', dir=ROOT / 'test')
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        (self.data / 'v1_1_train_nus.json').write_text(json.dumps([{'messages': []}] * 6))
        self.output = self.data / 'must_not_be_created'
        self.pointer = ROOT / 'test/output/latest_run.json'
        self.before = self.pointer.read_bytes() if self.pointer.exists() else None

    def assert_rejected(self, command, expected):
        completed = subprocess.run([sys.executable] + command, cwd=ROOT,
                                   capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(expected, completed.stdout + completed.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.before, self.pointer.read_bytes() if self.pointer.exists() else None)

    def test_renamed_demo_directory_is_not_full_data(self):
        self.assert_rejected(['tools/prepare_full_data.py', '--data_path', str(self.data),
                              '--dataset_id', 'renamed_business_data'], 'DriveLM 正式训练入口尚未完成')

    def test_container_rejects_before_image_device_and_output_changes(self):
        self.assert_rejected(['tools/container.py', 'full', '--data_path', str(self.data),
                              '--dataset_id', 'renamed_business_data',
                              '--output_dir', str(self.output)], 'DriveLM 正式训练入口尚未完成')

    def test_direct_model_entry_cannot_bypass_guard(self):
        self.assert_rejected(['tools/train_model.py', '--mode', 'full',
                              '--data_path', str(self.data), '--output_dir', str(self.output),
                              '--dataset_id', 'renamed_business_data'], 'DriveLM 正式训练入口尚未完成')

    def test_direct_upstream_launcher_cannot_bypass_guard(self):
        self.assert_rejected(['tools/run_train.py', 'qwen3vl8b', '--source', str(self.data),
                              '--devices', '0,1', '--output-dir', str(self.output),
                              '--model-path', str(self.data), '--dataset-id', 'renamed_business_data',
                              '--execute'], 'DriveLM 正式训练入口尚未完成')

    def test_annotation_check_rejects_demo_content(self):
        self.assert_rejected(['tools/prepare_full_data.py', '--data_path', str(self.data),
                              '--verify-annotations-only'], '缺少完整官方 DriveLM')


if __name__ == '__main__':
    unittest.main()
