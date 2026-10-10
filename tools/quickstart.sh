#!/usr/bin/env bash
# Run a pinned 1# training image with two physical NPUs and the selected model.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash tools/quickstart.sh {bevformer|sparse4d|qwen3vl8b} {check|plan|mini|train}

  check  Verify the selected image, packages and two NPU devices.
  plan   Show the mini training command and missing data without starting it.
  mini   Run a short model training check (BEVFormer: one mini epoch).
  train  Run the full training configuration; provide the full licensed data first.

Optional: ASCEND_HOST_DEVICES=0,2 (two different physical cards on 1#).
          MIGRATION_MOUNT_ROOT=/path/containing/migration/and/data
          QWEN_MODEL_DIR=/path/to/verified/Qwen3-VL-8B-Instruct
          QWEN_DATASET_ID=registered_dataset_name (required for Qwen train).
EOF
}

if [[ $# -ne 2 ]]; then usage; exit 2; fi
model=$1
action=$2
case "$action" in check|plan|mini|train) ;; *) usage; exit 2 ;; esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
mount_root=${MIGRATION_MOUNT_ROOT:-$(dirname "$repo_root")}
mount_root=$(realpath "$mount_root")
case "$repo_root/" in "$mount_root/"*) ;; *) echo "Repository must be under MIGRATION_MOUNT_ROOT: $mount_root" >&2; exit 2 ;; esac

case "$model" in
  bevformer|sparse4d)
    image=local/migration-drivingsdk:26.1.0-a3-headless-20261010
    image_id=sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
    py=/opt/conda/envs/$model/bin/python
    network=host
    memory=48g
    ;;
  qwen3vl8b)
    image=local/llamafactory:a3-arm64-20261010
    image_id=sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8
    py=/usr/local/python3.12.13/bin/python
    network=none
    memory=64g
    ;;
  *) usage; exit 2 ;;
esac

actual_id=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null) || {
  echo "Image $image is missing. See docs/environment.md." >&2
  exit 2
}
if [[ "$actual_id" != "$image_id" ]]; then
  echo "Image ID mismatch for $image: $actual_id (expected $image_id)" >&2
  exit 2
fi

IFS=, read -r -a host_ids <<< "${ASCEND_HOST_DEVICES:-0,2}"
if [[ ${#host_ids[@]} -ne 2 || ! ${host_ids[0]} =~ ^[0-9]+$ ||
      ! ${host_ids[1]} =~ ^[0-9]+$ || ${host_ids[0]} == "${host_ids[1]}" ]]; then
  echo "ASCEND_HOST_DEVICES must contain two distinct numeric host device IDs" >&2
  exit 2
fi

docker_args=(run --rm --network "$network" --memory "$memory" --cpus 8
             --pids-limit 1024 --shm-size 8g)
for id in "${host_ids[@]}"; do
  path=/dev/davinci$id
  if [[ ! -c "$path" ]]; then echo "Missing host NPU: $path" >&2; exit 2; fi
  docker_args+=(--device "$path")
done
for path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ ! -c "$path" ]]; then echo "Missing Ascend device: $path" >&2; exit 2; fi
  docker_args+=(--device "$path")
done
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware; do
  if [[ ! -e "$path" ]]; then echo "Missing Ascend host path: $path" >&2; exit 2; fi
  docker_args+=(-v "$path:$path:ro")
done
for path in /usr/local/bin/npu-smi /usr/local/dcmi /usr/bin/hccn_tool; do
  if [[ -e "$path" ]]; then docker_args+=(-v "$path:$path:ro"); fi
done
docker_args+=(-v "$mount_root:$mount_root:rw" -w "$repo_root"
              -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2
              -e HCCL_CONNECT_TIMEOUT=120 -e PYTHONUNBUFFERED=1)
if [[ "$model" == qwen3vl8b ]]; then
  docker_args+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
                -e TOKENIZERS_PARALLELISM=false)
else
  docker_args+=(-e "PATH=/opt/conda/envs/$model/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
fi
docker_args+=(--entrypoint "$py" "$image")

if [[ "$action" == check ]]; then
  command=(tools/check_env.py --model "$model" --nproc 2
           --output "outputs/$model/environment.json")
else
  command=(tools/run_train.py "$model" --devices 0,1
           --output-dir "outputs/$model")
  case "$model" in
    bevformer|sparse4d)
      source_name=BEVFormer
      [[ "$model" == sparse4d ]] && source_name=Sparse4D
      command+=(--source "sources/$source_name" --sdk sources/DrivingSDK)
      if [[ "$action" == mini || "$action" == plan ]]; then
        command+=(--smoke --dataset-version v1.0-mini)
      fi
      ;;
    qwen3vl8b)
      command+=(--source sources/LLaMA-Factory
                --model-path "${QWEN_MODEL_DIR:-$mount_root/weights/qwen3-vl-8b-instruct}"
                --dataset-id "${QWEN_DATASET_ID:-mllm_demo}")
      if [[ "$action" == mini || "$action" == plan ]]; then command+=(--smoke); fi
      ;;
  esac
  if [[ "$action" == mini || "$action" == train ]]; then command+=(--execute); fi
fi

echo "Image: $image ($image_id)"
echo "Host NPUs: /dev/davinci${host_ids[0]}, /dev/davinci${host_ids[1]} -> container IDs 0,1"
echo "Action: $model $action"
exec docker "${docker_args[@]}" "${command[@]}"
