# BEVFormer-base：Ascend 训练迁移

## 参考实现与范围

上游是 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) 的 `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`。复用 [DrivingSDK BEVFormer 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md) 的 NPU 补丁，以 FP16 base 配置做 2 卡训练。`sources.lock.json` 固定了上游、DrivingSDK 和补丁哈希。

## 环境和数据

在隔离的 DrivingSDK/BEVFormer 环境中安装配套 CANN、PyTorch、TorchNPU、mx_driving，以及按官方补丁源码构建的 MMCV、mmdet、mmdet3d。训练前执行 `python3 tools/check_env.py --model bevformer --nproc 2`。版本配套与镜像候选见仓库 `docs/environment.md`。

模型源码下需有 nuScenes v1.0 full、CAN bus、`data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl`、`ckpts/r101_dcn_fcos3d_pretrain.pth`。请按 DrivingSDK 样例生成标注；数据和权重不进入 Git。

公开的 nuScenes `v1.0-mini` 可用于有限步功能试验。用[官方 mini 归档](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)解压到数据根目录后，将源码目录的 `data/nuscenes` 链接到该目录，使用上游 `tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data` 生成标注。若没有单独的 CAN bus 扩展包，上游转换器要求 `data/can_bus` 目录存在，且会将缺失场景的 CAN bus 信息填零；这种标注仅用于功能检查，不用于精度结论。预训练权重仍须按上游 README 获取。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只检查并打印启动计划：

```bash
python3 tools/prepare.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --apply
bash PyTorch/contrib/cv/detection/BEVFormer/test/train_2p.sh --smoke
bash PyTorch/contrib/cv/detection/BEVFormer/test/train_2p.sh --smoke --execute
bash PyTorch/contrib/cv/detection/BEVFormer/test/train_2p.sh --smoke --dataset-version v1.0-mini --execute
```

FP16 base 的 8 卡基线学习率为 `2e-4`；2 卡且每卡 batch 1 时以 `5e-5` 为线性缩放起点。`--smoke` 对 epoch runner 执行 1 个 epoch 并跳过验证，可能需要较长时间。
`--dataset-version v1.0-mini` 仅允许与 `--smoke` 同用；启动前核对 `data/nuscenes/v1.0-mini/scene.json` 和生成的 pkl。mini 的 10 个场景不足以复现完整训练精度。

## 验证和状态

本地已验证补丁逐文件内容、Python 编译和命令生成；910C 训练尚未执行。实机需核对算子前后向、两 rank 有限 loss、checkpoint，以及 nuScenes mAP/NDS。学习率和精度需由实训确定。
