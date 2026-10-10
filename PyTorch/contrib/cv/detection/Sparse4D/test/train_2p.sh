#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
sources_dir=${MIGRATION_SOURCES_DIR:-$repo_root/sources}
python_bin=${PYTHON_BIN:-python3}
exec "$python_bin" "$repo_root/tools/run_train.py" sparse4d \
  --source "$sources_dir/Sparse4D" --sdk "$sources_dir/DrivingSDK" \
  --devices "${ASCEND_DEVICES:-0,1}" \
  --output-dir "$repo_root/outputs/sparse4d" "$@"
