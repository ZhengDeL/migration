# Apollo VisionNet：Ascend 训练迁移

## 范围与实测状态

采用 [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net/tree/4909aa681bc435a74e793e714b8066d9bbf0095d) 固定提交 `4909aa681bc435a74e793e714b8066d9bbf0095d`，迁移 `bev_tiny_det_occ_apollo.py` 的 DLA34 / 时序 BEV / GroupDETR 检测 / OCC 训练路径。

2026-10-11 在 1# 四张不同物理卡上，通过公开 quickstart 入口完成真实数据 10 步训练。完整 OCC 归档已经校验并安全解压。当前数据保留官方划分与 mini 的交集：242 train / 162 独立 val，原始 OCC 标签及非零 CAN bus 保留。使用官方仓内 DLA 深度预训练权重，训练前全部 222 个张量匹配；保存后对独立 val 的 162/162 帧完成推理有限值检查，**未计算正式精度指标**。

更正：此前“官方 DLA 权重未提供”的描述不准确，固定官方仓已在 `ckpts/` 提供该文件。本次已核验并重跑官方初始化训练；随机初始化的历史结果保留在归档中。

## 镜像

实际使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010` 的 `bevformer` 环境：Python 3.10.21、CANN 9.1.0、Torch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。官方来源、digest、实测 image ID 和构建方法见[环境指南](../../../../../docs/environment.md)。

## 在已准备好的 1# 上运行

从仓库根目录执行，一次运行一个模型：

```bash
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
bash tools/quickstart.sh apollo_visionnet verify
```

默认宿主逻辑设备 `0,2,4,6`，容器内为 `0,1,2,3`。运行前检查空闲；换卡示例：`ASCEND_HOST_DEVICES=8,10,12,14 bash tools/quickstart.sh apollo_visionnet mini`。

`mini` 从固定上游配置生成 FP32 四卡、每卡 batch 1、AdamW 学习率 `1e-4` 的十步配置。保留原始三帧 / 六相机、900×1600 图像读取及 0.5 缩放 / padding、50×50 BEV 和 640000 体素的 OCC 结构。短训练不运行指标评测，结束后另用 `verify` 检查全部独立验证帧；自动检查原始输出、掩码处理前 loss、梯度、参数更新、被检查参数的跨 rank 一致性和保存状态。

成功判据：命令退出码 0，`outputs/apollo_visionnet/mini/training_check_rank{0,1,2,3}.json` 均为 `pass_training_smoke`；rank 0 的 `checkpoint_reload` 为 `pass`、`optimizer_step_values` 为 `[10.0]`。保存文件 `iter_10.pth` 为 844544332 字节，SHA256 `f2b2c77e1b6df0a58ffcafff308b18a0fa5774d67dbd8efa7d284229569d4a29`；600 个模型张量 / 465 个优化器状态均有限。公开逐 rank 证据见[实测记录](../../../../../docs/validation.md)。这项保存检查未包含从 checkpoint 继续训练。

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
bash tools/quickstart.sh apollo_visionnet verify
```

`prepare` 校验本仓从官方文件观测并锁定的 PKL SHA256，选取 mini 对应帧，检查每个稀疏 OCC 标签的形状、整数索引和类别范围，复制原标签并记录 SHA256；生成 `data/apollo/mini_prepared/manifest.json`。这些 PKL 哈希是本次下载观测值，不能说成发布方公布的哈希。完整 train/val 标签模式已在 1# 实测：404 个标签逐文件检查并记录复制前后 SHA256，训练和验证 token 无交集，覆盖全部 mini 样本。

如未解压，将 `occ_gt_release_v1_0.tar.gz` 放到 `data/apollo/`，`prepare` 自动核对 15,840,721,593 字节与本次观测 SHA256 `52ae4e3637fc4c1b1aa2d1fb4273676c06606200566f3d436e2a359edd1b1282`，完整检查 gzip CRC、103302 个 tar 成员与安全路径后解压。该 SHA256 是用户提供文件的本次观测值，未取得发布方公布哈希；完整归档结构通过不表示所有 NPY 已作语义检查，本次逐数组验证范围为选出的 404 个标签。

### 官方初始化权重

固定官方源码自带 `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`。大小 61,045,957 字节，官方 Git blob 为 `af42f8693fd64f47021e64d6b530331cf075e901`，本次 SHA256 为 `8c55b0d98fd0fbc885a7487a764457941e008dcc5a76887196e6264a0e38cc1d`；配置生成时自动核验全部值。训练 hook 检查所有 222 个预训练张量与初始化后模型完全相同，证据见[最新训练记录](../../../../../verification/host1_20261011/apollo_official_init/rank0.json)。

## 适配内容

- `ascend.patch`：训练入口、NPU 分布式包装及三处多尺度变形注意力改为 DrivingSDK 实现。
- `compat.patch`：旋转角显式转为 Python `float`；文件句柄加载标注时指定 PKL 格式；NPU 上使用 MMDetection 已有的 `py_sigmoid_focal_loss`，保留原 gamma / alpha / weight / reduction / avg_factor 语义并注册 Apollo 损失映射。
- 原编译版 FocalLoss 在真实训练入口发生 `PyThreadState_Get` 致命崩溃。替换后的 FP32 CPU/NPU 前后向对比 8 个案例通过 `atol=1e-5, rtol=1e-4`，最大梯度绝对误差约 `6.20e-6`；累计 sum 的 loss 绝对误差可达 `0.001953125`，满足相对误差标准。证据见[算子对比 JSON](../../../../../verification/host1_20261011/apollo_real/focal_parity.json)。未进行 CUDA 基准或完整数据精度对齐。

`build` 只构建模型，`compute` 保留低分辨率合成计算检查。上游补丁哈希见仓库[来源锁定](../../../../../sources.lock.json)。

## 正式训练待办

正式完整训练需补齐完整 nuScenes 与 CAN bus，复用已校验的 OCC 归档及官方 DLA 权重，按上游目录准备后运行 `bash tools/quickstart.sh apollo_visionnet train`。正式入口会检查完整数据及权重路径；四卡每卡 batch 1、学习率 `1e-4` 为实验起点，需重新评估 batch / 学习率。mini 不替代检测 mAP、占用 mIoU、稳定性、性能和[官方训练硬件支持](../../../../../docs/environment.md)验收。
