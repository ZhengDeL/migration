#!/usr/bin/env bash
# Run a pinned 1# training image and the selected model diagnostic or training.
set -euo pipefail

usage() {
  cat <<'EOF'
用法：bash tools/quickstart.sh {bevformer|sparse4d|qwen3vl8b} {check|plan|mini|train}
      bash tools/quickstart.sh qwen3vl8b download
      bash tools/quickstart.sh apollo_visionnet {check|prepare|build|compute|plan|mini|train}
      bash tools/quickstart.sh worldengine {check|download|prepare|build|compute|plan|mini|train}

  check  检查镜像、软件包和对应数量的 NPU，不启动训练。
  plan   列出 mini 训练命令和缺失文件，不启动训练。
  mini   运行短程功能训练（BEVFormer 为 mini 数据 1 个 epoch）。
  train  使用完整训练配置；先准备有授权的完整数据。
  download  Qwen 下载完整固定权重；WorldEngine 下载约 5.59 GB 功能验证数据及词表，校验并准备子集。
  prepare   用已下载的官方文件准备 Apollo/WorldEngine 真实样本功能子集。
  build    构建 Apollo/WorldEngine 模型并放到 NPU；不加载训练数据或预训练权重。
  compute  Apollo/WorldEngine 四卡合成输入前后向与参数更新；不代表真实数据训练。

可选环境变量：ASCEND_HOST_DEVICES=0,2 （1# 两张不同物理卡的逻辑 ID）
              MIGRATION_MOUNT_ROOT=/同时包含仓库与数据的目录
              QWEN_MODEL_DIR=/已校验的模型目录
              QWEN_DATASET_ID=已注册的数据集名 （Qwen 正式训练必填）
              APOLLO_OCC_DIR=/解压后的/occ_gt_release_v1_0
              APOLLO_VAL_ONLY=1 （仅有 VAL 标签时复用做功能训练，不评估精度）
EOF
}

if [[ $# -ne 2 ]]; then usage; exit 2; fi
model=$1
action=$2
case "$action" in check|build|compute|plan|mini|train|download|prepare) ;; *) usage; exit 2 ;; esac
if [[ "$action" == download ]] && [[ "$model" != qwen3vl8b && "$model" != worldengine ]]; then
  echo "download 用于 qwen3vl8b 权重或 worldengine 功能验证数据。" >&2
  exit 2
fi
if [[ "$action" == build || "$action" == compute ]] && [[ "$model" != apollo_visionnet && "$model" != worldengine ]]; then
  echo "build/compute 用于 apollo_visionnet 和 worldengine；其余模型使用 check/mini。" >&2
  exit 2
fi
if [[ "$action" == prepare ]] && [[ "$model" != apollo_visionnet && "$model" != worldengine ]]; then
  echo "prepare 用于 apollo_visionnet 和 worldengine。" >&2
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
  worldengine)
    image=local/migration-worldengine:26.1.0-a3-20261011
    image_id=sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3
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

if [[ "$action" == download || "$action" == prepare ]]; then
  if [[ "$model" == qwen3vl8b ]]; then
    download_dir=$(realpath -m "${QWEN_MODEL_DIR:-$mount_root/weights/qwen3-vl-8b-instruct}")
    case "$download_dir" in "$mount_root/"*) ;; *)
      echo "QWEN_MODEL_DIR 必须位于 MIGRATION_MOUNT_ROOT 挂载目录下：$mount_root" >&2
      exit 2
      ;;
    esac
  fi
  download_args=(run --rm --network host --memory 16g --cpus 4
                 --pids-limit 512 -v "$mount_root:$mount_root:rw" -w "$repo_root"
                 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2)
  if [[ "$action" == prepare ]]; then
    download_args[3]=none
    if [[ "$model" == worldengine ]]; then
      exec docker "${download_args[@]}" --entrypoint "$py" "$image" \
        tools/prepare_world_mini.py --source sources/WorldEngine \
        --data-dir "$mount_root/data/worldengine_smoke"
    else
      prepare_args=(tools/prepare_apollo_mini.py --data-dir "$mount_root/data/apollo"
                    --occ-dir "${APOLLO_OCC_DIR:-$mount_root/data/apollo/occ_gt_release_v1_0}"
                    --nuscenes "$mount_root/data/nuscenes"
                    --prepared-dir "$mount_root/data/apollo/mini_prepared")
      if [[ ${APOLLO_VAL_ONLY:-0} == 1 ]]; then prepare_args+=(--val-only); fi
      exec docker "${download_args[@]}" --entrypoint "$py" "$image" "${prepare_args[@]}"
    fi
  fi
  https_proxy_value=${HTTPS_PROXY:-${https_proxy:-}}
  http_proxy_value=${HTTP_PROXY:-${http_proxy:-$https_proxy_value}}
  if [[ -n "$https_proxy_value" ]]; then
    download_args+=(-e "HTTPS_PROXY=$https_proxy_value" -e "https_proxy=$https_proxy_value")
  fi
  if [[ -n "$http_proxy_value" ]]; then
    download_args+=(-e "HTTP_PROXY=$http_proxy_value" -e "http_proxy=$http_proxy_value")
  fi
  if [[ "$model" == qwen3vl8b ]]; then
    echo "从 Qwen 官方 ModelScope 仓下载固定版本；完成后逐分片校验 SHA256。"
    exec docker "${download_args[@]}" --entrypoint "$py" "$image" \
      tools/fetch_qwen.py --hub modelscope --destination "$download_dir" --execute
  else
    echo "下载官方 NAVTEST 标注、PDM 缓存、三段日志的图像、感知基座权重及词表；完成后校验并准备功能训练子集。"
    exec docker "${download_args[@]}" --entrypoint "$py" "$image" \
      tools/fetch_world_mini.py --source sources/WorldEngine \
      --destination "$mount_root/data/worldengine_smoke" --execute
  fi
fi

IFS=, read -r -a host_ids <<< "${ASCEND_HOST_DEVICES:-$default_host_ids}"
if [[ ${#host_ids[@]} -ne $nproc ]]; then
  echo "$model 需要 $nproc 个宿主设备编号，如 $default_host_ids" >&2
  exit 2
fi
declare -A seen_ids=()
declare -A seen_physical=()
for id in "${host_ids[@]}"; do
  if [[ ! "$id" =~ ^[0-9]+$ ]]; then echo "NPU 设备编号必须是数字：$id" >&2; exit 2; fi
  if [[ -n ${seen_ids[$id]:-} ]]; then echo "NPU 设备编号重复：$id" >&2; exit 2; fi
  seen_ids[$id]=1
  physical=$((10#$id / 2))
  if [[ -n ${seen_physical[$physical]:-} ]]; then
    echo "1# 设备 $id 和 ${seen_physical[$physical]} 属于同一物理卡；请选择不同物理卡，如 $default_host_ids" >&2
    exit 2
  fi
  seen_physical[$physical]=$id
done
container_devices=$(seq -s, 0 $((nproc - 1)))
if [[ "$action" == mini || "$action" == train || "$action" == compute || "$action" == build ]]; then
  python3 "$repo_root/tools/check_idle_cards.py" "$(IFS=,; echo "${host_ids[*]}")"
fi

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
  source_name=Apollo-Vision-Net
  [[ "$model" == worldengine ]] && source_name=WorldEngine
  command=(tools/check_model_build.py --model "$model" --source "sources/$source_name"
           --output "outputs/$model/model_build.json")
elif [[ "$action" == compute ]]; then
  compute_tool=tools/check_apollo_compute.py
  source_name=Apollo-Vision-Net
  if [[ "$model" == worldengine ]]; then
    compute_tool=tools/check_world_compute.py
    source_name=WorldEngine
  fi
  command=(-m torch.distributed.run --standalone --nproc_per_node="$nproc"
           "$compute_tool" --source "sources/$source_name"
           --output-dir "outputs/$model/compute_fixture")
else
  output_dir="outputs/$model"
  if [[ "$model" == apollo_visionnet || "$model" == worldengine ]] && [[ "$action" == mini || "$action" == plan ]]; then
    output_dir="outputs/$model/mini"
  fi
  command=(tools/run_train.py "$model" --devices "$container_devices"
           --output-dir "$output_dir")
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
      if [[ "$action" == mini || "$action" == plan ]]; then
        command+=(--smoke --data-dir "$mount_root/data/apollo"
                  --prepared-dir "$mount_root/data/apollo/mini_prepared"
                  --nuscenes "$mount_root/data/nuscenes")
      fi
      ;;
    worldengine)
      command+=(--source sources/WorldEngine)
      if [[ "$action" == mini || "$action" == plan ]]; then
        command+=(--smoke --data-dir "$mount_root/data/worldengine_smoke"
                  --prepared-dir "$mount_root/data/worldengine_smoke/prepared")
      fi
      ;;
  esac
  if [[ "$action" == mini || "$action" == train ]]; then command+=(--execute); fi
fi

echo "镜像：$image ($image_id)"
echo "宿主 NPU：${host_ids[*]} -> 容器编号 $container_devices"
echo "操作：$model $action"
exec docker "${docker_args[@]}" "${command[@]}"
