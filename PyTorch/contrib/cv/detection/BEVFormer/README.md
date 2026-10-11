# BEVFormer-base：Ascend 工程使用指南

文档复核：2026-10-12；实机验证：2026-10-10/11。按官方的环境、数据、训练和结果检查顺序组织，使用本仓实测 NPU 入口。

## 1. 模型与环境

| 项目 | 本仓采用值 |
| --- | --- |
| 官方实现 | [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer/tree/66b65f3a1f58caf0507cb2a971b9c0e7f842376c)，固定提交 `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` |
| 昇腾参考 | [DrivingSDK BEVFormer 样例](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/BEVFormer/README.md) |
| 模型配置 | `projects/configs/bevformer_fp16/bevformer_base_fp16.py`，R101-DCN / FP16 base |
| 资源 | ARM64 Linux，2 张空闲物理卡；1# 默认宿主逻辑 ID `0,2` |
| 镜像 / 环境 | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` / `bevformer` |
| 软件栈 | Python 3.10.21 / CANN 9.1.0 / Torch 2.7.1+cpu / TorchNPU 2.7.1.post6 / mx_driving 26.1.0.post2 |

镜像 ID、OpenMMLab 配套版本及官方硬件支持边界见[环境指南](../../../../../docs/environment.md)。启动器核对 image ID，容器提供训练依赖。

## 2. 已有交付资产：检查并运行

新机器先完成[离线恢复](../../../../../docs/offline_delivery.md)。所有命令在迁移仓根目录运行，恢复后替换示例 `cd` 路径。重新训练前按[保存旧结果](../../../../../docs/quickstart.md#重新训练前保存旧结果)保留 `outputs/bevformer/`。

```bash
cd /var/ai-model/lzd/migration
npu-smi info
bash tools/quickstart.sh bevformer check
bash tools/quickstart.sh bevformer plan
bash tools/quickstart.sh bevformer mini
```

`check` 检查两个 NPU 和环境；`plan` 须显示 `missing_data: []`；`mini` 运行 **1 epoch**，本次为 162 次迭代，耗时长于十步检查。默认两卡每卡 batch 1、学习率 `5e-5`，跳过指标评测。换卡示例：`ASCEND_HOST_DEVICES=8,10 bash tools/quickstart.sh bevformer mini`；先核对实际物理映射。

## 3. 从零准备源码、数据与权重

在仓库根目录下载固定源码并核验适配：

```bash
python3 tools/fetch_sources.py drivingsdk bevformer --execute
python3 tools/prepare.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --apply
```

数据要求遵循[官方数据说明](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/docs/prepare_dataset.md)。本仓源码内的必要布局：

```text
sources/BEVFormer/
├── data/can_bus/
├── data/nuscenes/
│   ├── samples/  sweeps/  maps/
│   ├── v1.0-mini/
│   ├── nuscenes_infos_temporal_train.pkl
│   └── nuscenes_infos_temporal_val.pkl
└── ckpts/r101_dcn_fcos3d_pretrain.pth
```

离线交付已具备这些 mini 资产。自行准备时，先把 `data/nuscenes` 链接到解压数据，建立 `data/can_bus`，取得官方初始化权重。**以下转换命令在镜像的 bevformer Python 环境内、迁移仓根目录执行**，不能直接使用宿主的任意 Python：

```bash
python3 tools/create_bevformer_mini_infos.py \
  --source sources/BEVFormer \
  --mmdet3d-dataset-source /workspace/bevformer/mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py \
  --canbus-root sources/BEVFormer/data
```

此入口执行固定上游的元数据转换函数，避开 CPU 准备阶段的训练算子导入。本次未提供 CAN bus 扩展，404 条记录按上游逻辑填零，仅用于功能验证。正式训练需完整 `v1.0-trainval`、CAN bus 和重新生成的 temporal PKL，转换与下载方法见上述官方指南。

## 4. 成功判据与输出

| 检查 | 通过条件 / 结果位置 |
| --- | --- |
| 环境 | `outputs/bevformer/environment.json` 为 `pass`；管理面告警处理见[快速上手](../../../../../docs/quickstart.md) |
| 训练 | 退出码 0，完成 1 epoch，日志 loss 有限 |
| 保存 | `outputs/bevformer/epoch_1.pth` 可加载，不能只检查文件存在 |
| 本次保存实测 | epoch=1、iter=162，1089 模型张量 / 557 优化器状态有限，优化器 step=162 |

本次 checkpoint SHA 和原始证据见[验证记录](../../../../../docs/validation.md)，新训练的文件 SHA 可以不同。此模型没有 `quickstart verify` 操作。mini 未验收 mAP / NDS、性能、长期稳定性或继续训练。

## 5. 正式训练与常见问题

完整资产准备后执行 `bash tools/quickstart.sh bevformer train`。本仓正式入口为两卡每卡 batch 1、学习率 `5e-5`、24 epoch；学习率由官方八卡 `2e-4` 按 batch 缩放，是实验起点，需完整 nuScenes 指标验收。

| 提示 / 现象 | 处理 |
| --- | --- |
| `missing_data` 非空 | 补齐打印的 temporal PKL、原始图像、CAN bus 目录及权重，再运行 `plan` |
| data worker 报 `dict_keys` 序列化错误 | 使用固定源码及 `prepare.py --apply`；本仓 `compat.patch` 保持类别顺序并改为列表 |
| OpenCV / `libGL.so.1` 导入错误 | 导入已验证的 headless 派生镜像并核对 ID |
| 训练只出现 loss，未正常结束 | 检查最后异常和保存内容，不能认定通过 |

高级 `test/train_2p.sh` 需在正确容器环境中设置进程可见的 `ASCEND_DEVICES`；常规操作使用 `quickstart.sh`。补丁哈希见[sources.lock.json](../../../../../sources.lock.json)。
