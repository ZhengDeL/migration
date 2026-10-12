#!/usr/bin/env bash
set -euo pipefail
model_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$model_root/test/env_npu.sh"
exec python3 "$model_root/tools/container.py" plan "$@"
