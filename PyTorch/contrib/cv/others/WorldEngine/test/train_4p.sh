#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
sources_dir=${MIGRATION_SOURCES_DIR:-$repo_root/sources}
python_bin=${PYTHON_BIN:-python3}
: "${ASCEND_DEVICES:?set four logical NPU IDs after checking physical-card topology with npu-smi}"
exec "$python_bin" "$repo_root/tools/run_train.py" worldengine \
  --source "$sources_dir/WorldEngine" \
  --devices "$ASCEND_DEVICES" \
  --output-dir "$repo_root/outputs/worldengine" "$@"
