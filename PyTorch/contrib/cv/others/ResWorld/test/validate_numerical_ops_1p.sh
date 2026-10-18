#!/usr/bin/env bash
set -euo pipefail
model_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec python3 "$model_root/tools/container.py" numerical --nproc 1 "$@"
