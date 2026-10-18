# MobileNetV2 for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

图像分类基础网络。采用华为 Ascend ModelZoo 的 MobileNetV2，保留 width_mult=1.0 和原始 1000 类输出头，作为智驾视觉骨干的迁移参考。

## 参考实现

- 参考实现：

  ```text
  url=https://gitcode.com/Ascend/ModelZoo-PyTorch.git
  commit_id=e779117e8cfeba296a7105e26adac0741ab04768
  code_path=PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch
  ```

- 原仓训练／数据说明：[固定版本文档](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

固定原文将该模型列为“不随版本演进模型”，其历史八卡/Apex 指标属于原发布环境。本项目的 Torch2.7.1、两卡 FP32 和当前 V1 权重评测分别有实机记录。

## 迁移说明

沿用 ModelZoo 网络；只将 url.ini 的读取路径固定到网络文件旁，确保更换工作目录仍能读取。训练使用显式 NPU Tensor、原生 HCCL DDP、FP32 SGD；功能运行 lr=0.001、momentum=0.9、weight_decay=4e-5。

默认功能运行：每 rank batch=4，lr=0.001，双卡各执行 10 次优化器更新。10 次更新是整个 DDP 训练的 10 个全局 step，不相加成 20 步。

# 准备训练环境

## 安装昇腾环境

ARM64 Linux、Docker、兼容宿主驱动与两张空闲物理卡；先核对宿主 `npu-smi info`。

## 安装模型环境

实际验证机器为用户提供的 1# ARM64 Atlas 800I A3，使用 2 张物理卡；默认宿主逻辑 ID 为 `0,2`。A3 一张物理卡对应两个逻辑 ID，本项目每张物理卡只选一个。Atlas 800I 属于推理系列，这次实验不改变官方训练硬件支持范围。

| 项目 | 本次实测 |
| --- | --- |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| 镜像 ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| 官方基镜像 manifest | `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` |
| 配套 | CANN 9.1.0 / Python 3.10.21 / PyTorch 2.7.1+cpu / TorchNPU 2.7.1.post6 / DrivingSDK 26.1.0.post2 |
| 容器资源 | 主机内存上限 48 GiB，8 CPU，8 GiB shm |
| 精度与通信 | FP32 / HCCL DDP |

镜像继承官方 DrivingSDK，额外使用无显示依赖的 OpenCV 配套。`images.lock.json` 固定实际 image ID 和离线归档 hash。工程不会因上游出现新 tag 自动更换已经验证的镜像。

## 准备数据集

fastai 官方 Imagenette-160 公开图片，按固定文件排序选取 100 张 train 和 40 张 val。10 个 WNID 映射到标准 ImageNet 1000 类索引，未把输出头改成 10 类；训练和验证图片内容 SHA256 不重叠。

`bash test/setup.sh` 使用本模型 `data/raw/imagenette2-160.tgz` 和 `imagenet_class_index.json`，生成 `data/functional/`。训练采用 RandomResizedCrop(224) 与随机翻转，验证采用 Resize(256) / CenterCrop(224)，归一化均保持 ImageNet 均值和方差。

数据及输出路径必须在本模型目录内，便于整个目录迁移。复用已经准备的数据时先核对 hash；出现摘要不一致不要覆盖旧数据。

## 准备预训练或交付权重

功能训练为独立训练状态；官方完整评测使用 `data/imagenet_official/mobilenet_v2-b0353104.pth`，SHA256、官方 val/devkit 归档与预处理见 [完整验证指南](FULL_VALIDATION.md#4-官方-imagenet-50000-张完整验证)。该 V1 权重来自 ModelZoo 原 `url.ini`，不能将 Imagenette 功能权重的指标记作 V1 完整评测。

# 开始训练

## 训练模型

适用条件：ARM64 Linux Ascend 服务器，Docker、宿主驱动及 Python 3 可用，两张物理卡空闲。使用交付的本模型独立 ZIP，解压后进入含本 README 的目录；目录内已包含镜像、数据与检查点。GitHub 仅存放代码和指导，运行所需大文件由独立离线包提供。



以下为十步功能入口。进入实际解压目录后顺序执行：

```bash
python3 tools/verify_package.py
bash test/setup.sh
bash test/check_env.sh
bash test/plan.sh
bash test/train_2p.sh
bash test/validate_1p.sh
```



`setup` 校验并导入缺失的镜像，然后准备本模型数据；镜像已存在时核对 image ID。`plan` 只核对数据和输出运行参数，不代表训练完成。训练成功后会更新 `test/output/latest_run.json`，验证在独立容器中严格重载最近一次成功训练的 checkpoint。原有交付检查点可直接执行 `bash test/validate_1p.sh --output_dir "$PWD/test/output/delivery_verified"` 复验。

每次 NPU 容器退出后，设备释放可能需要几秒。入口记录本模型上一容器退出时间，连续运行自动等待最多 8 秒；出现 Device Resource Busy，先确认上一次本模型容器已退出并稍后重试。脚本检测卡被占用会报错，工程入口不会擅自停止其他业务服务。

## 单卡、换卡与历史功能续训

```bash
# 更换两张空闲物理卡的宿主逻辑 ID
bash test/train_2p.sh --devices 4,6
# 使用单张物理卡做功能训练
bash test/train_1p.sh --devices 0
# 继续训练同一数据 / 同一 rank 数的 checkpoint，steps 为目标累计更新数
bash test/train_2p.sh --resume "$PWD/test/output/delivery_verified/checkpoint.pt" --steps 20
```



新 schema2 检查点恢复权重、优化器、epoch/batch 游标和全部 rank Python/NumPy/Torch/NPU RNG，并校验精度、batch 和学习率。显式 FP32 配置下已完成 10＋10 与连续20步控制，478 个状态张量全部逐位相同。旧 schema1 只支持历史权重/优化器续训，不承诺精确重放。指定输出目录必须不存在。

## 完整数据训练

将有授权的完整 ImageNet-2012 放到本模型 `data/imagenet_original/`；train / val 都按全部 1000 个 WNID 分类目录存放。

```bash
bash test/prepare_data.sh --mode full --imagenet_path "$PWD/data/imagenet_original"
bash test/train_full_2p.sh
```

入口核对 train=1281167、val=50000、1000 类目录和内容不重叠。完整模式默认 600 epoch、batch=256 每 rank、SGD lr=0.05（两卡 global batch=512 的本项目配置），沿用 ModelZoo 每 2.5 epoch 乘 0.98 的 step 学习率公式。FP32 原生 SGD / HCCL 与 ModelZoo 八卡 / Apex 配置有区别，完整训练及精度尚未验证；不可沿用官方原始精度结论。Imagenette 功能数据会被 full 守卫拒绝。

## 独立验证

完整解压本模型最新 ZIP 后，在本目录执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

该入口自动校验完整文件清单、准备固定镜像，并将随包检查点复制到新的输出目录完成独立 NPU 重载与真实数据验证。结果路径由命令输出给出。重新训练及当前验收边界见 [完整验证指南](FULL_VALIDATION.md)。


扩展数值验证、显式 FP32 配置和官方 ImageNet 全集评测入口见 [完整验证指南](FULL_VALIDATION.md)。本页原指标为首次功能基线。

## 官方预训练权重完整 ImageNet 评测

先按本模型 FULL_VALIDATION.md 核对 V1 权重和官方 50,000 图 val / devkit 归档；随后执行：

```bash
bash test/validate_imagenet_1p.sh --devices 0 \
  --output_dir "$PWD/test/output/imagenet_v1_full_validation"
```

完整评测比较两个后端全量 logits 与 Top-1/Top-5。它与 600 epoch 从头训练分别记录。

# 训练结果展示

## 本项目实测（2026-10-18）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），驱动 26.1.1 / 固件 9.0.0.9.220；固定镜像及软件版本见本页环境表。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 功能训练数据 | Imagenette：100 train / 40 val；保留 1,000 类输出头；2 张物理卡、全局 batch=8、20 步控制 |
| 正式验证数据 | ImageNet-2012 全部 50,000 val；官方 V1 预训练权重，非本项目从头训练权重 |
| 全量分类指标 | CPU 与 NPU：Top-1=35,935/50,000=71.87%；Top-5=45,147/50,000=90.294% |
| CPU/NPU 全量输出 | 50,000,000 个 logits，超差 0；最大绝对误差 6.8664551e-5；完整判据见全量 JSON |
| 梯度 | 固定源权重、新 20 步权重各 158/158 有效梯度通过；整体相对 L2 分别 1.1653562e-4 / 4.0335445e-4，上限 0.01 |
| 严格续训 | 478/478 完整状态张量逐位相同；最大误差 0；真实增强后输入 SHA 一致 |
| 持续训练 | 双卡 1,800.501374 s / 新增 33,570 次更新；独立重载 CPU/NPU max=2.4795532e-5 |
| 未完成 | 完整 ImageNet 600 epoch 从头训练；驾驶检测、分割和规划业务指标 |

当前判定：`pass_official_pretrained_full_validation_and_functional_training_checks`。

当前主要权重 SHA256：`b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`。用途：ImageNet 全量评测使用的官方 V1 预训练权重；功能训练与续训使用另存的本项目检查点。

## 训练性能实测

| 进程数 | 全局吞吐（图像/s） | 每 rank batch | 预热更新 | 计时更新 |
| --- | --- | --- | --- | --- |
| 1p | 87.506 | 4 | 10 | 50 |
| 2p | 153.102 | 4 | 10 | 50 |

实测包含真实数据加载、传输、前向、反向及优化器更新，各更新同步；使用不同物理卡的单个逻辑芯片。主机 CPU/存储有其他任务共享，没有提供生产性能目标；本表不作为官方 FPS 或性能达标证明。

## 验证依据与复现

[逐项实机结果及原门槛](validation/extended_acceptance_20261018.json)记录实际执行阶段、退出码、输入范围、权重身份、数值/恢复/持续训练及未完成项。复现命令和结果文件见[完整训练与验证指导](FULL_VALIDATION.md)。旧十步基线保留在 `validation/train_result.json` 和 `validation/validation_result.json`，不能覆盖本节当前结果。

恢复测试比较完整模型、优化器及适用的调度器、训练游标和各 rank RNG；单纯能加载模型不等于续训通过。CPU/NPU 使用相同输入与相同权重的数值对照，与数据集任务指标分别报告。

# 版本说明

## 变更

- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示／现象 | 处理 |
| --- | --- |
| 只有 GitHub 源码，缺少镜像或数据 | 取得本模型完整 ZIP，按本页固定镜像和数据清单核对 |
| image ID 或数据摘要不符 | 对照本模型 images.lock.json 和来源锁；使用匹配版本 |
| Device Resource Busy | 核对物理拓扑与占用，等待本模型容器释放后重试 |
| 输出目录已存在 | 为新运行指定本模型内不存在的目录，保留旧结果 |
| 需要精确续训 | 按完整指南选择正确检查点，保持数据、rank、batch 和数值配置一致 |
| 仅有日志没有通过结果 | 核对退出码及本次 JSON，不能将旧结果当本次通过 |

## 许可

原始 MobileNetV2 文件包含 Huawei / Soumith BSD-3-Clause 版权说明，完整许可保留于 vendor/LICENSE.MobileNetV2。本项目新增脚本采用 Apache-2.0。
