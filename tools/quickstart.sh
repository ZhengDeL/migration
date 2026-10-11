#!/usr/bin/env bash
# Run a pinned 1# training image and the selected model diagnostic or training.
set -euo pipefail

usage() {
  cat <<'EOF'
用法：bash tools/quickstart.sh {bevformer|sparse4d|qwen3vl8b} {check|plan|mini|train}
      bash tools/quickstart.sh qwen3vl8b download
      bash tools/quickstart.sh apollo_visionnet {check|build|compute|plan|train}

  check  检查镜像、软件包和两张 NPU，不启动训练。
  plan   列出 mini 训练命令和缺失文件，不启动训练。
  mini   运行短程功能训练（BEVFormer 为 mini 数据 1 个 epoch）。
  train  使用完整训练配置；先准备有授权的完整数据。
  download  从 Qwen 官方 ModelScope 仓下载固定版本权重并校验 SHA256。
  build    构建 Apollo 模型并放到 NPU；不加载数据或预训练权重。
  compute  Apollo 四卡合成输入前后向与参数更新检查；不代表真实数据训练。

可选环境变量：ASCEND_HOST_DEVICES=0,2 （1# 两张不同物理卡的逻辑 ID）
              MIGRATION_MOUNT_ROOT=/同时包含仓库与数据的目录
              QWEN_MODEL_DIR=/已校验的模型目录
              QWEN_DATASET_ID=已注册的数据集名 （Qwen 正式训练必填）
EOF
}

if [[ $# -ne 2 ]]; then usage; exit 2; fi
model=$1
action=$2
case "$action" in check|build|compute|plan|mini|train|download) ;; *) usage; exit 2 ;; esac
if [[ "$action" == download && "$model" != qwen3vl8b ]]; then
  echo "download 目前只用于 qwen3vl8b" >&2
  exit 2
fi
if [[ "$action" == build || "$action" == compute ]] && [[ "$model" != apollo_visionnet ]]; then
  echo "build/compute 当前用于 apollo_visionnet；其余模型使用 check/mini。" >&2
  exit 2
fi
if [[ "$model" == apollo_visionnet && "$action" == mini ]]; then
  echo "Apollo 尚无已准备好的真实 OCC mini 数据；先运行 compute 检查四卡计算路径。" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
mount_root=${MIGRATION_MOUNT_ROOT:-$(dirname "$repo_root")}
mount_root=$(realpath "$mount_root")
case "$repo_root/" in "$mount_root/"*) ;; *) echo "仓库必须位于 MIGRATION_MOUNT_ROOT 之下：$mount_root" >&2; exit 2 ;; esac

case "$model" in
  bevformer|sparse4d)
    image=local/migration-drivingsdk:26.1.0-a3-headless-20261010
    image_id=sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
    py=/opt/conda/envs/$model/bin/python
    network=host
    memory=48g
    nproc=2
    default_host_ids=0,2
    model_env=$model
    ;;
  apollo_visionnet)
    image=local/migration-drivingsdk:26.1.0-a3-headless-20261010
    image_id=sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
    model_env=bevformer
    py=/opt/conda/envs/bevformer/bin/python
    network=host
    memory=96g
    nproc=4
    default_host_ids=0,2,4,6
    ;;
  qwen3vl8b)
    image=local/llamafactory:a3-arm64-20261010
    image_id=sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8
    py=/usr/local/python3.12.13/bin/python
    network=none
    memory=64g
    nproc=2
    default_host_ids=0,2
    ;;
  *) usage; exit 2 ;;
esac

actual_id=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null) || {
  echo "缺少镜像 $image；请查看 docs/environment.md。" >&2
  exit 2
}
if [[ "$actual_id" != "$image_id" ]]; then
  echo "镜像 ID 不匹配：$image 实际 $actual_id，预期 $image_id" >&2
  exit 2
fi

if [[ "$action" == download ]]; then
  download_dir=$(realpath -m "${QWEN_MODEL_DIR:-$mount_root/weights/qwen3-vl-8b-instruct}")
  case "$download_dir" in "$mount_root/"*) ;; *)
    echo "QWEN_MODEL_DIR 必须位于 MIGRATION_MOUNT_ROOT 挂载目录下：$mount_root" >&2
    exit 2
    ;;
  esac
  download_args=(run --rm --network host --memory 16g --cpus 4
                 --pids-limit 512 -v "$mount_root:$mount_root:rw" -w "$repo_root"
                 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0)
  https_proxy_value=${HTTPS_PROXY:-${https_proxy:-}}
  http_proxy_value=${HTTP_PROXY:-${http_proxy:-$https_proxy_value}}
  if [[ -n "$https_proxy_value" ]]; then
    download_args+=(-e "HTTPS_PROXY=$https_proxy_value" -e "https_proxy=$https_proxy_value")
  fi
  if [[ -n "$http_proxy_value" ]]; then
    download_args+=(-e "HTTP_PROXY=$http_proxy_value" -e "http_proxy=$http_proxy_value")
  fi
  echo "从 Qwen 官方 ModelScope 仓下载固定版本；完成后逐分片校验 SHA256。"
  exec docker "${download_args[@]}" --entrypoint "$py" "$image" \
    tools/fetch_qwen.py --hub modelscope \
    --destination "$download_dir" --execute
fi

IFS=, read -r -a host_ids <<< "${ASCEND_HOST_DEVICES:-$default_host_ids}"
if [[ ${#host_ids[@]} -ne $nproc ]]; then
  echo "$model 需要 $nproc 个宿主设备编号，如 $default_host_ids" >&2
  exit 2
fi
declare -A seen_ids=()
for id in "${host_ids[@]}"; do
  if [[ ! "$id" =~ ^[0-9]+$ ]]; then echo "NPU 设备编号必须是数字：$id" >&2; exit 2; fi
  if [[ -n ${seen_ids[$id]:-} ]]; then echo "NPU 设备编号重复：$id" >&2; exit 2; fi
  seen_ids[$id]=1
done
container_devices=$(seq -s, 0 $((nproc - 1)))

docker_args=(run --rm --network "$network" --memory "$memory" --cpus 8
             --pids-limit 1024 --shm-size 8g)
for id in "${host_ids[@]}"; do
  path=/dev/davinci$id
  if [[ ! -c "$path" ]]; then echo "宿主机缺少 NPU 设备：$path" >&2; exit 2; fi
  docker_args+=(--device "$path")
done
for path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ ! -c "$path" ]]; then echo "宿主机缺少 Ascend 设备：$path" >&2; exit 2; fi
  docker_args+=(--device "$path")
done
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware; do
  if [[ ! -e "$path" ]]; then echo "宿主机缺少 Ascend 目录：$path" >&2; exit 2; fi
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
  docker_args+=(-e "PATH=/opt/conda/envs/$model_env/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
fi
if [[ "$action" == check ]]; then docker_args+=(-e PYTHONWARNINGS=ignore); fi
docker_args+=(--entrypoint "$py" "$image")

if [[ "$action" == check ]]; then
  command=(tools/check_env.py --model "$model" --nproc "$nproc"
           --output "outputs/$model/environment.json" --summary)
elif [[ "$action" == build ]]; then
  command=(tools/check_model_build.py --model "$model" --source sources/Apollo-Vision-Net
           --output "outputs/$model/model_build.json")
elif [[ "$action" == compute ]]; then
  command=(-m torch.distributed.run --standalone --nproc_per_node="$nproc"
           tools/check_apollo_compute.py --source sources/Apollo-Vision-Net
           --output-dir "outputs/$model/compute_fixture")
else
  command=(tools/run_train.py "$model" --devices "$container_devices"
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
    apollo_visionnet)
      command+=(--source sources/Apollo-Vision-Net)
      if [[ "$action" == plan ]]; then command+=(--smoke); fi
      ;;
  esac
  if [[ "$action" == mini || "$action" == train ]]; then command+=(--execute); fi
fi

echo "镜像：$image ($image_id)"
echo "宿主 NPU：${host_ids[*]} -> 容器编号 $container_devices"
echo "操作：$model $action"
exec docker "${docker_args[@]}" "${command[@]}"
