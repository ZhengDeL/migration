#!/usr/bin/env bash
set -euo pipefail
MODEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$MODEL_ROOT/tools/container.py" full --nproc 1 "$@"
