# Sparse4Dv3：Ascend 训练迁移

## 参考实现与范围

上游是 [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 的 `c41df4bbf7bc82490f11ff55173abfcb3fb91425`。复用 [DrivingSDK Sparse4D 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md) 的模型和依赖补丁。`tools/prepare.py` 会核验官方 `tools/patch.py`，并生成两处可审计的本仓覆盖：把 `clip_grad_norm_fused_` 改为 `torch.nn.utils.clip_grad_norm_`，并保留配置中的普通 `AdamW`，不强制改成 `NpuFusedAdamW`；其余字节保持一致。1# 首次双卡训练在融合裁剪处出现梯度与掩码长度不一致，两个 rank 均报 `EZ1007`；改用逐参数裁剪后，融合 AdamW 的参数更新仍有长度不一致。逐参数裁剪保留原来的 `max_norm=25` 和 2 范数；普通优化器的性能需另行评估。官方原文件的 SHA256 为 `f2742e70b7dc06e50849c95eb8deb45911d4ca4bee92e35393f8d09185c933cf`，覆盖后的 SHA256 为 `d0a89e3e4c775b0b3a51893c4bbfe2c9483bfc89d4ce4d947b9bb506d7f2c791`。

## 环境和数据

使用隔离的 DrivingSDK/Sparse4D 环境，按官方样例构建 NPU 版 MMCV、mmdet 并安装 mx_driving。训练前执行 `python3 tools/check_env.py --model sparse4d --nproc 2`。需准备 nuScenes、`data/nuscenes_anno_pkls/nuscenes_infos_{train,val}.pkl`、`nuscenes_kmeans900.npy` 和 ResNet50 预训练权重；数据制作按官方样例执行。

[DrivingSDK 示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md)给出的 `resnet50-19c8e357.pth` 使用旧 tar 序列化，PyTorch 2.7.1 的安全加载不能直接读取。`sources.lock.json` 固定官方 URL 和源文件 SHA256；下载后用 `tools/convert_sparse4d_weight.py` 核对哈希，生成 `ckpt/resnet50-19c8e357-modern.pth`，同时逐 tensor 核对新旧文件。转换过程必须在无网络、无 NPU、只暴露权重输出目录的隔离容器中执行；训练配置副本引用新文件，保留官方原文件。示例从本仓根目录运行：

```bash
docker run --rm --network none --memory 3g --pids-limit 512 \
  -v "$PWD":/work:ro -v "$PWD/sources/Sparse4D/ckpt":/weights:rw -w /work \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  --entrypoint /opt/conda/envs/sparse4d/bin/python \
  local/drivingsdk:26.1.0-a3-arm64-20261010 \
  tools/convert_sparse4d_weight.py \
  --input /weights/resnet50-19c8e357.pth \
  --output /weights/resnet50-19c8e357-modern.pth --execute
```

用公开 `v1.0-mini` 做功能试验时，将源码目录的 `data/nuscenes` 链接到解压后的 mini 根目录，在 Sparse4D 源码目录运行 `python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix data/nuscenes_anno_pkls/nuscenes`，再按[DrivingSDK 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md)运行 `tools/anchor_generator.py` 生成 `nuscenes_kmeans900.npy`。mini 生成的锚框只用于功能检查，正式训练须用完整数据重新生成。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只打印计划：

```bash
python3 tools/prepare.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --apply
bash PyTorch/contrib/cv/detection/Sparse4D/test/train_2p.sh --smoke
bash PyTorch/contrib/cv/detection/Sparse4D/test/train_2p.sh --smoke --execute
bash PyTorch/contrib/cv/detection/Sparse4D/test/train_2p.sh --smoke --dataset-version v1.0-mini --execute
```

启动器由官方配置生成输出目录中的副本，把 `total_batch_size` 48→2、`num_gpus` 8→2、学习率 `6e-4`→`2.5e-5`，并引用上述转换后的权重，同时按 2 卡重新计算每 epoch 迭代数。`--smoke` 只跑 10 次优化迭代，不覆盖上游配置。
mini 模式另外将配置副本中的数据集版本改为 `v1.0-mini`，不会改写上游配置；只允许与 `--smoke` 同用。

## 验证和状态

本地已验证补丁逐文件内容、配置生成、Python 编译和命令生成；1# 双卡 mini 训练的融合梯度裁剪及融合 AdamW 首次更新均已复现形状错误。两处覆盖后，双卡 mini 完成 10 迭代并正常退出，10 条 loss 均为有限值（首条 37.1234，末条 41.4400；短程内并非单调下降）。`iter_10.pth` 可加载，含 819 个模型张量和 652 项优化器状态。该结果是功能初验；完整数据集 nuScenes mAP/NDS、普通 AdamW 相比融合优化器的性能与正式硬件支持尚未验收，学习率以精度结果确认。
