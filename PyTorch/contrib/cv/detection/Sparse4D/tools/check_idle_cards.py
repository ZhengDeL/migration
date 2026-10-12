#!/usr/bin/env python3
# Model-local implementation; no other model directory is required.
"""Read-only guard for this task's selected physical NPU cards."""
import subprocess
import sys
physical_ids = sorted({int(value) // 2 for value in sys.argv[1].split(',')})
output = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=20)
for index in physical_ids:
    if 'No running processes found in NPU ' + str(index) + ' ' not in output:
        raise SystemExit('Selected physical NPU is not idle: ' + str(index))
print('Selected physical NPU cards are idle:', physical_ids)
