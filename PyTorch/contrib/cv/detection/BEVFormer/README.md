# BEVFormer-base：Ascend 训练迁移

## 参考实现与范围

上游是 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) 的 `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`。复用 [DrivingSDK BEVFormer 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md) 的 NPU 补丁，以 FP16 base 配置做 2 卡训练。`sources.lock.json` 固定了上游、DrivingSDK 和补丁哈希。1# 首次模型启动在数据 worker 的 `spawn` 阶段因 `eval_detection_configs.class_names` 是不可序列化的 `dict_keys` 而失败；本仓 `compat.patch` 将其转为保持类别顺序的 `list`，并由 `tools/prepare.py` 与官方补丁一起逐文件核验。

## 环境和数据

1# 实际双卡 mini 训练使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010`（Docker image ID `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`）的独立 `bevformer` 环境。该镜像从华为 DrivingSDK A3/openEuler 官方标签 `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10` 的 ARM64 manifest `sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` 派生，只修正预置 OpenCV 的 headless 冲突；构建方法见仓库[环境指南](../../../../../docs/environment.md)和[Dockerfile](../../../../../docker/DrivingSDK-headless.Dockerfile)。实测环境为 CANN 9.1.0、Python 3.10.21、PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post6、mx_driving 26.1.0.post2，并预置 BEVFormer 对应的 MMCV/mmdet/mmdet3d。训练前执行 `python3 tools/check_env.py --model bevformer --nproc 2`；镜像导入不等于当前设备已通过训练检查。

模型源码下需有 nuScenes v1.0 full、CAN bus、`data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl`、`ckpts/r101_dcn_fcos3d_pretrain.pth`。请按 DrivingSDK 样例生成标注；数据和权重不进入 Git。

公开的 nuScenes `v1.0-mini` 可用于有限步功能试验。用[官方 mini 归档](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)解压到数据根目录后，将源码目录的 `data/nuscenes` 链接到该目录，并创建 `data/can_bus` 目录。上游 `tools/create_data.py` 在转换前就导入训练算子，纯 CPU 数据准备会因 mx_driving 查询设备而失败。本仓的 `tools/create_bevformer_mini_infos.py` 从固定的上游文件执行原始元数据函数，从镜像内 mmdet3d 源码读取类别映射，跳过训练算子和不参与 BEVFormer 训练的 2D 导出。在仓库根目录、DrivingSDK 的 `bevformer` 环境执行：

```bash
python3 tools/create_bevformer_mini_infos.py \
  --source sources/BEVFormer \
  --mmdet3d-dataset-source /workspace/bevformer/mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py \
  --canbus-root sources/BEVFormer/data
```

若没有单独的 CAN bus 扩展包，固定上游转换函数会将缺失场景的 CAN bus 信息填零；这种标注仅用于功能检查，不用于精度结论。预训练权重仍须按上游 README 获取。

## 快速开始

1# 上已准备镜像和 mini 数据时，从仓库根目录依次执行 `bash tools/quickstart.sh bevformer check`、`bash tools/quickstart.sh bevformer plan`、`bash tools/quickstart.sh bevformer mini`。命令作用和通过标准见[快速上手](../../../../../docs/quickstart.md)。

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

本地已验证补丁逐文件内容、Python 编译和命令生成。1# 的双卡基础训练与变形注意力数值检查通过；`compat.patch` 应用后，双卡 nuScenes mini 完成 162 迭代/1 epoch 并正常退出。训练日志第 50/100/150 迭代 loss 分别为 23.3374/19.9635/18.4445，均为有限值；`epoch_1.pth` 可加载，元数据为 epoch 1、iter 162，含 1089 个模型张量和 557 项优化器状态。该结果是功能初验；完整数据集 nuScenes mAP/NDS、稳定性和正式硬件支持尚未验收，学习率与精度需由正式训练确定。
