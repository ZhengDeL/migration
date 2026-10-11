# Apollo VisionNet：Ascend 训练迁移

## 范围与实测状态

采用 [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net/tree/4909aa681bc435a74e793e714b8066d9bbf0095d) 固定提交 `4909aa681bc435a74e793e714b8066d9bbf0095d`，迁移 `bev_tiny_det_occ_apollo.py` 的 DLA34 / 时序 BEV / GroupDETR 检测 / OCC 训练路径。

2026-10-11 在 1# 四张不同物理卡上，通过公开 quickstart 入口完成真实数据 10 步训练。当前 162 帧来自官方 VAL 与 nuScenes mini 的交集，原始 OCC 标签和 CAN bus 保留；VAL 仅复用做功能训练，**不评估独立验证精度**。深度预训练 DLA34 权重未提供，短训练明确使用随机初始化。

## 镜像

实际使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010` 的 `bevformer` 环境：Python 3.10.21、CANN 9.1.0、Torch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。官方来源、digest、实测 image ID 和构建方法见[环境指南](../../../../../docs/environment.md)。

## 在已准备好的 1# 上运行

从仓库根目录执行，一次运行一个模型：

```bash
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
```

默认宿主逻辑设备 `0,2,4,6`，容器内为 `0,1,2,3`。运行前检查空闲；换卡示例：`ASCEND_HOST_DEVICES=8,10,12,14 bash tools/quickstart.sh apollo_visionnet mini`。

`mini` 从固定上游配置生成 FP32 四卡、每卡 batch 1、AdamW 学习率 `1e-4` 的十步配置。保留原始三帧 / 六相机、900×1600 图像读取及 0.5 缩放 / padding、50×50 BEV 和 640000 体素的 OCC 结构。禁用验证，仅检查功能；自动检查原始输出、掩码处理前 loss、梯度、参数更新、被检查参数的跨 rank 一致性和保存状态。

成功判据：命令退出码 0，`outputs/apollo_visionnet/mini/training_check_rank{0,1,2,3}.json` 均为 `pass_training_smoke`；rank 0 的 `checkpoint_reload` 为 `pass`、`optimizer_step_values` 为 `[10.0]`。保存文件 `iter_10.pth` 为 844544012 字节，SHA256 `5288adfd7555740da0a41ba04fd2bad30d866a6b9e628fe948d4bf628b576c3d`；600 个模型张量 / 465 个优化器状态均有限。公开逐 rank 证据见[实测记录](../../../../../docs/validation.md)。这项保存检查未包含从 checkpoint 继续训练。

## 在其他机器准备 mini 数据

按固定上游[官方数据说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/prepare_dataset.md)下载并解压有权使用的数据，默认布局：

```text
/var/ai-model/lzd/data/nuscenes/v1.0-mini/sample.json
/var/ai-model/lzd/data/nuscenes/samples/...
/var/ai-model/lzd/data/apollo/nuscenes_infos_temporal_train_occ_gt.pkl
/var/ai-model/lzd/data/apollo/nuscenes_infos_temporal_val_occ_gt.pkl
/var/ai-model/lzd/data/apollo/occ_gt_release_v1_0/train/...
/var/ai-model/lzd/data/apollo/occ_gt_release_v1_0/val/...
```

```bash
bash tools/quickstart.sh apollo_visionnet prepare
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
```

`prepare` 校验本仓从官方文件观测并锁定的 PKL SHA256，选取 mini 对应帧，检查每个稀疏 OCC 标签的形状、整数索引和类别范围，复制原标签并记录 SHA256；生成 `data/apollo/mini_prepared/manifest.json`。这些 PKL 哈希是本次下载观测值，不能说成发布方公布的哈希。完整 train/val 标签模式尚待完整 OCC 数据补齐后实测。

只有 VAL 标签时，可显式加 `APOLLO_VAL_ONLY=1`；OCC 在其他路径时加 `APOLLO_OCC_DIR=/解压后的/occ_gt_release_v1_0`。当前 1# 的 162 个完整 VAL 标签来自下载到的官方归档前段，逐成员检查通过，但 **15.84 GB 完整归档未校验成功**。Google Drive 返回下载额度耗尽，完整 OCC 数据等待补齐。禁止将前段文件当成完整数据或生成替代标签。

## 适配内容

- `ascend.patch`：训练入口、NPU 分布式包装及三处多尺度变形注意力改为 DrivingSDK 实现。
- `compat.patch`：旋转角显式转为 Python `float`；文件句柄加载标注时指定 PKL 格式；NPU 上使用 MMDetection 已有的 `py_sigmoid_focal_loss`，保留原 gamma / alpha / weight / reduction / avg_factor 语义并注册 Apollo 损失映射。
- 原编译版 FocalLoss 在真实训练入口发生 `PyThreadState_Get` 致命崩溃。替换后的 FP32 CPU/NPU 前后向对比 8 个案例通过 `atol=1e-5, rtol=1e-4`，最大梯度绝对误差约 `6.20e-6`；累计 sum 的 loss 绝对误差可达 `0.001953125`，满足相对误差标准。证据见[算子对比 JSON](../../../../../verification/host1_20261011/apollo_real/focal_parity.json)。未进行 CUDA 基准或完整数据精度对齐。

`build` 只构建模型，`compute` 保留低分辨率合成计算检查。上游补丁哈希见仓库[来源锁定](../../../../../sources.lock.json)。

## 正式训练待办

补齐完整 nuScenes、CAN bus、OCC 归档和官方 DLA34 深度预训练权重，按上游目录准备后运行 `bash tools/quickstart.sh apollo_visionnet train`。正式入口会检查完整数据及权重路径；四卡每卡 batch 1、学习率 `1e-4` 为实验起点，需重新评估 batch / 学习率。mini 不替代检测 mAP、占用 mIoU、稳定性、性能和[官方训练硬件支持](../../../../../docs/environment.md)验收。
