#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
sources_dir=${MIGRATION_SOURCES_DIR:-$repo_root/sources}
python_bin=${PYTHON_BIN:-python3}
: "${ASCEND_DEVICES:?set two logical NPU IDs after checking physical-card topology with npu-smi}"
exec "$python_bin" "$repo_root/tools/run_train.py" qwen3vl8b \
  --source "$sources_dir/LLaMA-Factory" \
  --devices "$ASCEND_DEVICES" \
  --output-dir "$repo_root/outputs/qwen3vl8b" \
  --model-path "${QWEN_MODEL_DIR:-$sources_dir/Qwen3-VL-8B-Instruct}" \
  --dataset-id "${QWEN_DATASET_ID:-mllm_demo}" "$@"
