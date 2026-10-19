#!/usr/bin/env python3
"""Prepare official DriveLM media, train/dev rows, validation questions and context audit."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    for name in ['prepare_drivelm_complete.py', 'prepare_drivelm_validation.py', 'audit_drivelm_context.py']:
        subprocess.run([sys.executable, str(ROOT / 'tools' / name)], cwd=ROOT, check=True)
    print('Complete DriveLM preparation and every-row context audit passed; no NPU training was run.')
