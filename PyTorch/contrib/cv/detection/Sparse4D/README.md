# Sparse4Dv3：Ascend 工程使用指南

文档复核：2026-10-12；实机验证：2026-10-10/11。按官方的环境、数据、训练和结果检查顺序组织，使用本仓实测两卡 NPU 入口。

## 1. 模型与环境

| 项目 | 本仓采用值 |
| --- | --- |
| 官方实现 | [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D/tree/249ffbb695f4e9db628d953e2bf6d36de04bbb69)，固定提交 `249ffbb695f4e9db628d953e2bf6d36de04bbb69` |
| 昇腾参考 | [DrivingSDK Sparse4D 样例](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/Sparse4D/README.md) |
| 配置 | `projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py`，Sparse4Dv3 / ResNet50 |
| 资源 | ARM64 Linux，2 张空闲物理卡；1# 默认逻辑 ID `0,2` |
| 镜像 / 环境 | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` / `sparse4d` |
| 软件栈 | Python 3.10.21 / CANN 9.1.0 / Torch 2.7.1+cpu / TorchNPU 2.7.1.post6 / mx_driving 26.1.0.post2 |

镜像 ID、配套 MMCV/mmdet 及硬件支持边界见[环境指南](../../../../../docs/environment.md)。固定源码相比 SDK 基线仅新增 README 两行，运行代码相同，补丁已核验。

## 2. 已有交付资产：检查并运行

新机器先完成[离线恢复](../../../../../docs/offline_delivery.md)。以下均在迁移仓根目录运行，恢复后替换示例 `cd` 路径。重新训练前按[保存旧结果](../../../../../docs/quickstart.md#重新训练前保存旧结果)保留 `outputs/sparse4d/`。

```bash
cd /var/ai-model/lzd/migration
npu-smi info
bash tools/quickstart.sh sparse4d check
bash tools/quickstart.sh sparse4d plan
bash tools/quickstart.sh sparse4d mini
```

`plan` 须显示 `missing_data: []`。mini 是 **10 次优化迭代**；两卡每卡 batch 1、普通 AdamW、学习率 `2.5e-5`。配置副本写入 outputs，原配置保留。

## 3. 从零准备源码、数据与权重

在迁移仓根目录：

```bash
python3 tools/fetch_sources.py drivingsdk sparse4d --execute
python3 tools/prepare.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --apply
```

遵循[官方 Quick Start](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/docs/quick_start.md)与上述 SDK 样例。必要布局：

```text
sources/Sparse4D/
├── data/nuscenes/
├── data/nuscenes_anno_pkls/
│   ├── nuscenes_infos_train.pkl
│   └── nuscenes_infos_val.pkl
├── nuscenes_kmeans900.npy
└── ckpt/
    ├── resnet50-19c8e357.pth
    └── resnet50-19c8e357-modern.pth
```

交付已包含 mini 标注、锚框和两份权重。自行生成时，先链接原始 nuScenes 数据，**在镜像的 sparse4d 环境内、Sparse4D 源码根目录**执行：

```bash
cd sources/Sparse4D
mkdir -p data/nuscenes_anno_pkls
python3 tools/nuscenes_converter.py --version v1.0-mini --info_prefix data/nuscenes_anno_pkls/nuscenes
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 tools/anchor_generator.py \
  --ann_file data/nuscenes_anno_pkls/nuscenes_infos_train.pkl \
  --output_file_name nuscenes_kmeans900.npy
```

显式指定锚框文件名，避免上游命令行默认的 `_nuscenes_kmeans900.npy` 与本训练配置不一致。正式训练用完整 trainval/test 重新生成 PKL 和锚框。

官方 ResNet50 文件采用旧 tar 序列化，PyTorch 2.7.1 安全加载不能直接读取。本仓锁定原文件 SHA，转换后逐张量核对。自行下载后，在**迁移仓根目录**用无网络、无 NPU 容器转换：

```bash
docker run --rm --network none --memory 3g --pids-limit 512 \
  -v "$PWD":/work:ro -v "$PWD/sources/Sparse4D/ckpt":/weights:rw -w /work \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  --entrypoint /opt/conda/envs/sparse4d/bin/python \
  local/migration-drivingsdk:26.1.0-a3-headless-20261010 \
  tools/convert_sparse4d_weight.py --input /weights/resnet50-19c8e357.pth \
  --output /weights/resnet50-19c8e357-modern.pth --execute
```

## 4. 成功判据与输出

| 检查 | 通过条件 / 结果位置 |
| --- | --- |
| 环境 | `outputs/sparse4d/environment.json`；管理面告警处理见[快速上手](../../../../../docs/quickstart.md) |
| 训练 | 退出码 0，日志有 10 条有限 loss；短程 loss 不要求单调下降 |
| 保存 | `outputs/sparse4d/iter_10.pth` 可加载；本次 819 模型张量 / 652 优化器状态有限，优化器 step=10 |
| 计数 | SDK `meta.iter=9` 为零基计数，不能据此判断少跑了一步 |

本次 checkpoint SHA 见[验证记录](../../../../../docs/validation.md)，新训练的 SHA 可以不同。此模型没有 `quickstart verify` 操作。

## 5. 正式训练、适配与常见问题

本仓保留普通 AdamW 和逐参数梯度裁剪 `max_norm=25`、2 范数。1# 上官方辅助脚本强制的融合裁剪与融合 AdamW 均发生长度不一致；`prepare.py` 核验原文件并应用两处覆盖，其他模型与依赖补丁沿用固定 SDK 文件。

完整资产准备后执行 `bash tools/quickstart.sh sparse4d train`。两卡总 batch 2、学习率 `2.5e-5`、100 epoch；按卡数重新计算每 epoch 迭代数。学习率由上游 batch 48 的 `6e-4` 缩放，是实验起点，需完整 mAP / NDS、稳定性和性能验收。

| 提示 / 现象 | 处理 |
| --- | --- |
| 锚框文件缺失 | 检查 `--output_file_name` 与 `plan` 打印路径 |
| 旧权重加载失败 | 用上述转换入口，保留官方原文件，不修改全局加载行为 |
| 融合裁剪 / 优化器形状错误 | 核验 `prepare.py --apply` 的两处覆盖 |
| 卡重复或不空闲 | 核对宿主物理映射，换用空闲的不同物理卡 |

mini 未覆盖收敛、正式精度、性能或续训。固定提交与补丁哈希见[sources.lock.json](../../../../../sources.lock.json)。
