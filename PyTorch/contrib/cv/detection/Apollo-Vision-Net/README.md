# Apollo VisionNet：Ascend 工程使用指南

文档复核：2026-10-12；实机验证：2026-10-11。覆盖官方 DLA 初始化、四卡 mini 训练和保存后全部独立验证帧的推理功能检查。

## 1. 模型与环境

| 项目 | 本仓采用值 |
| --- | --- |
| 官方实现 | [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net/tree/4909aa681bc435a74e793e714b8066d9bbf0095d)，固定提交 `4909aa681bc435a74e793e714b8066d9bbf0095d` |
| 官方说明 | [数据准备](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/prepare_dataset.md)、[训练和测试](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/run.md) |
| 配置 | `projects/configs/bevformer/bev_tiny_det_occ_apollo.py`，DLA34 / 时序 BEV / GroupDETR / OCC |
| 资源 | ARM64 Linux；训练 4 张空闲物理卡，验证推理 1 张 |
| 镜像 / 环境 | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` / `bevformer` |
| 软件栈 | Python 3.10.21 / CANN 9.1.0 / Torch 2.7.1+cpu / TorchNPU 2.7.1.post6 / mx_driving 26.1.0.post2 |

image ID 为 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。来源、MMCV / mmdet / mmdet3d 配套和硬件支持边界见[环境指南](../../../../../docs/environment.md)。

## 2. 已有交付资产：检查、训练、推理

新机器先完成[离线恢复](../../../../../docs/offline_delivery.md)。以下均在迁移仓根目录运行，恢复后替换示例 `cd` 路径。已有 checkpoint 可先 verify；重新训练前按[保存旧结果](../../../../../docs/quickstart.md#重新训练前保存旧结果)保留 `outputs/apollo_visionnet/`。

```bash
cd /var/ai-model/lzd/migration
npu-smi info
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
bash tools/quickstart.sh apollo_visionnet verify
```

plan 必须显示 `missing_data: []`。mini 用 FP32、四卡每卡 batch 1、AdamW 学习率 `1e-4`、10 步；保留三帧 / 六相机、900×1600 原图与 0.5 缩放/padding、50×50 BEV 和 640000 体素 OCC 结构。

训练默认宿主逻辑 ID `0,2,4,6`，verify 默认只用 `0`。换卡分别设置 `ASCEND_HOST_DEVICES=8,10,12,14` 和 `ASCEND_HOST_DEVICES=8`。verify 严格重载保存模型，检查全部 **162 个独立验证帧**，不计算 mAP / mIoU。

## 3. 从零准备源码、数据与权重

在迁移仓根目录：

```bash
python3 tools/fetch_sources.py apollo_visionnet --execute
python3 tools/prepare.py apollo_visionnet --source sources/Apollo-Vision-Net --apply
```

数据遵循官方准备说明。mini 功能准备在仓库父目录使用：

```text
data/nuscenes/
├── v1.0-mini/sample.json
└── samples/...
data/apollo/
├── nuscenes_infos_temporal_train_occ_gt.pkl
├── nuscenes_infos_temporal_val_occ_gt.pkl
├── occ_gt_release_v1_0.tar.gz          # 或已解压的同名目录
└── mini_prepared/                     # 由 prepare 生成
```

将完整 OCC 原归档和官方 train/val PKL 放到对应位置，nuScenes mini 图像准备完毕，再运行：

```bash
bash tools/quickstart.sh apollo_visionnet prepare
bash tools/quickstart.sh apollo_visionnet plan
```

首次 OCC 解压需预留至少 100 GB 数据空间；整套交付恢复需 300 GB 可用空间。脚本核对归档 15,840,721,593 字节、观测 SHA、gzip CRC、103302 个 tar 成员和安全路径；已解压可复用。不同 OCC 路径用 `APOLLO_OCC_DIR=/实际路径/occ_gt_release_v1_0`。

prepare 保留官方划分与 mini 的交集：**242 train / 162 独立 val**，token 无交集，原始 OCC 标签和非零 CAN bus 保留；逐数组检查 404 个 mini 标签，并记录复制 SHA。完整归档结构检查不等于全部 NPY 的语义验证。归档与 PKL 的 SHA 是本次观测值，未取得发布方公布哈希。

官方初始化权重已经在固定源码的 `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`；61,045,957 字节，Git blob、SHA 和实际初始化全部核验。mini 训练开始前 222/222 个初始化张量匹配，证据见[官方初始化报告](../../../../../verification/host1_20261011/apollo_official_init/rank0.json)。

## 4. 成功判据与输出

| 检查 | 通过条件 / 文件 |
| --- | --- |
| mini | 退出码 0；`outputs/apollo_visionnet/mini/training_check_rank{0,1,2,3}.json` 全部为 `pass_training_smoke` |
| 计算内容 | 原始输出、掩码前 loss、465 梯度有限；参数更新和被检查参数跨 rank 一致 |
| 保存 | rank 0 的 `checkpoint_reload` 为 `pass`、`optimizer_step_values` 为 `[10.0]` |
| checkpoint | `outputs/apollo_visionnet/mini/iter_10.pth`；本次 600 模型张量 / 465 优化器状态有限 |
| verify | `outputs/apollo_visionnet/mini/heldout_check.json` 为 `pass_heldout_functional`，`frames_checked: 162` |

本次 checkpoint SHA 见[验证记录](../../../../../docs/validation.md)，新训练 SHA 可以不同。保存重载检查不包含续训；输出有限不等于正式精度通过。

## 5. 适配、正式训练与常见问题

NPU 补丁覆盖分布式入口和变形注意力；兼容补丁处理 NumPy 旋转角、PKL 文件句柄格式和 FocalLoss。NPU 使用 MMDetection 的 PyTorch focal 公式，保留 gamma/alpha/weight/reduction/avg_factor；8 个 FP32 CPU/NPU 前后向案例通过 `atol=1e-5, rtol=1e-4`，不是完整模型 CUDA 精度对齐。

完整训练还需完整 nuScenes/CAN bus，按官方结构准备源码内 `data/occ_gt_release_v1_0/` 下的 train/val PKL 和标签路径，然后运行 `bash tools/quickstart.sh apollo_visionnet train`。四卡每卡 batch 1、学习率 `1e-4`、24 epoch 是实验起点，需检测 mAP、占用 mIoU、稳定性和性能验收。

| 提示 / 现象 | 处理 |
| --- | --- |
| 缺少 OCC / PKL / 图像 | 按官方说明补齐，再 prepare、plan；不生成假标签 |
| 只有 VAL 标签 | 历史 `APOLLO_VAL_ONLY=1` 模式复用 VAL 作训练，不满足独立验证；当前交付使用完整 train/val |
| 官方初始化核验失败 | 检查固定源码和权重 SHA，不改为随机初始化绕过门禁 |
| verify 设备数错误 | 单卡操作只填一个宿主逻辑 ID |
| 只通过 build / compute | build 是构建检查、compute 是合成输入检查，仍需 mini 和独立帧 verify |

来源及补丁哈希见[sources.lock.json](../../../../../sources.lock.json)，完整验收边界见[环境指南](../../../../../docs/environment.md)。
