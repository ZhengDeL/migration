# ResWorld for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

采用[作者固定源码](https://github.com/mengtan00/ResWorld/tree/a5ae973001dd6001d95b3389020a9e263d73a56e)，保留80,766,238参数、三时刻六相机、原深度与两项规划loss。具体来源与原始文件摘要见本模型来源锁。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/mengtan00/ResWorld
  commit_id=a5ae973001dd6001d95b3389020a9e263d73a56e
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/README.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

## 迁移说明

| 文件 | 实际修改 |
| --- | --- |
| npu_activations.py | 仅原三个GELU使用官方NPU erf接口，CPU默认erf保持；执行计数、数值profile校验 |
| temporal_residual.py | CPU/NPU以S@(F0-F1)等价替换S@F0-S@F1，保留重复位置量零导数连接 |
| camera_geometry.py | 原get_sample_coor与原归一化固定CPU FP32；仅常量几何，拒绝可学习输入；网格传回NPU |
| prepare_source.py | vendor字节/SHA保持，只在runtime副本打可追踪补丁；文件锁保护多rank准备 |
| npu_compat.py | 原可训练池化、原MMCV NPU autograd；no_grad保留DrivingSDK；CPU add/relu公式兼容 |
| npu_precision.py / deterministic_communication.py | 关闭HF32；HCCL确定性与同步mean allreduce |
| training_state.py / train.py / resume_control.py | 全rank RNG及游标、增强前跳过已消费批次、输入SHA与三个数值profile |

相机标定原FP64矩阵计算和固定相机投影属于无梯度元数据处理；NN参数、特征、loss、有效梯度和采样反向均在NPU。Torch基础wheel的`+cpu`版本标签不代表NN在CPU执行，实际设备和调用守卫已实测。

根因和原失败对照见 [数值修复说明](NUMERICAL_REPAIR.md)。

# 准备训练环境

## 安装昇腾环境

ARM64 Linux、Docker、兼容宿主 Ascend 驱动；训练需要四张空闲物理卡。Atlas 800I A3 为本项目实验机器，官方训练支持范围不由本次实验改变。每物理卡只选一个逻辑 ID。

## 安装模型环境

镜像：`local/migration-drivingsdk:26.1.0-a3-headless-20261010`。

精确image ID：`sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。

CANN9.1.0、Python3.10.21、Torch2.7.1+cpu、TorchNPU2.7.1.post6、DrivingSDK26.1.0.post2；MMCV1.7.2/MMDet2.24/MMDet3D1.0.0rc4。离线镜像归档、SHA和来源见 `images.lock.json`。宿主需要Docker、兼容Ascend驱动与空闲物理卡；实际板卡为Atlas800I A3推理系列，实测不改变官方训练支持范围。

```bash
bash test/setup.sh
bash test/check_env.sh
```

`setup` 核对固定镜像和本模型原始数据，在本目录建立 runtime 副本及预处理数据；不启动训练。

## 准备数据集和预训练权重

数据来源和布局遵循[作者固定 README](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/README.md)的 nuScenes 准备步骤。离线包内的真实 mini 为 323/81 标注帧、8/2 互斥场景、20 个 CAN 文件和 2,424 个投影深度文件；验证使用全部 65 个完整 future 有效帧。

```text
data/nuscenes/                  # mini 图像、标注、CAN 与投影深度，setup 核验
ckpts/resworld-official-final.pth
ckpts/geobev-r50-nuimage-cbgs.pth
```

作者最终权重用于原 65 帧 CPU/NPU 规划与碰撞对照；GeoBEV 权重对应原仓初始化要求。当前 mini 功能训练从头运行，未将原 12 epoch / GeoBEV 初始化训练记作已完成。原权重完整性及本次数据摘要见本模型来源锁与验证 JSON。

# 开始训练

## 训练模型

本项目四卡真实 mini 功能入口；每次使用新输出目录：

```bash
bash test/setup.sh
bash test/check_env.sh
bash test/train_4p.sh --devices 0,2,4,6 --steps 10 --output_dir "$PWD/test/output/train_new"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/train_new"
```

每物理卡只选一个宿主逻辑ID；0,2,4,6代表四张不同物理卡。每次使用新输出目录。精确恢复、五份权重数值检查、吞吐与30分钟脚本均独立放在本模型目录。旧数值profile不能直接作为新profile精确续训。

## 独立验证

完整解压一个ResWorld.zip，进入本README目录，在ARM64 Linux Ascend服务器执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

自动校验完整包、准备固定镜像，在新目录重载随包最终权重。镜像、源码、真实数据和权重均在本模型ZIP内。GitHub保存源码、指导和验证摘要。重载检查的具体范围与完整训练/数值复验命令见[本模型指南](FULL_VALIDATION.md)。

## 数值、续训、吞吐与持续训练

依次按 [本模型完整验证指南](FULL_VALIDATION.md)执行算子检查、五份固定权重梯度、四卡 10+10/连续20控制、吞吐、30分钟和作者原65帧。各步骤使用自己的新输出目录，检查点明确指定，训练与评测记录三个数值 profile。

## 正式完整训练范围

作者原指导依次准备 nuScenes trainval、CAN、地图／深度、GeoBEV 初始化，运行四卡训练与最终评估。本项目正式 trainval / 原 12 epoch runner、EMA、SyncBN 和 warmup 协议尚未完成；`full` 入口继续拒绝。当前可复现范围为本页列明的真实 mini 工程验证。

# 训练结果展示

## 本项目实测（2026-10-18）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），驱动 26.1.1 / 固件 9.0.0.9.220；固定镜像及软件版本见本页环境表。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据与协议 | 真实 nuScenes mini：323 train / 81 val 标注帧，8/2 场景；CAN 20 文件、深度 2,424 文件 |
| 功能训练 | 4 张物理卡；FP32；真实增强数据 20 步控制，之后另做持续训练；不是原 12 epoch 全量训练 |
| 五份权重数值回归 | 原失败、后续失败、旧长稳、新 20 步、最终长稳权重各 446/446 有效梯度通过；原门槛保持 |
| 最终权重梯度 | 整体相对 L2=2.83294570892e-5，上限 0.01；逐参数、loss 及一步参数门槛均通过 |
| CPU 参考检查 | 常量几何投影修复前后 446/446 个 CPU 参考梯度逐位相同；时序等价化范围另见数值说明 |
| 严格续训 | 四卡 10+10 对连续 20：2,054/2,054 完整状态张量逐位相同，max=0；四 rank 增强输入 SHA 匹配 |
| 持续训练及重载 | 四卡 1,811.967955 s / 新增 660 次更新；最终累计 step=680；独立重载 CPU/NPU max=3.8146973e-6 |
| 作者权重评测 | 原规划/碰撞函数全部 65 帧 CPU/NPU 对照通过；轨迹 rtol=atol=0.003，逐帧离散碰撞完全一致 |
| 未完成 | 正式 nuScenes trainval、原 12 epoch / EMA / SyncBN / warmup；作者权重可能见过 mini，不作为独立驾驶精度验收 |

当前判定：`pass_mini_numerical_and_operational_gates_full_protocol_pending`。

当前主要权重 SHA256：`24e405e0f15156e7618a697be665d95bc53e5ef856431e2a426ae4287a505d3f`。用途：本项目最终持续训练模型及优化器状态，累计 step=680。

## 训练性能实测

| 进程数 | 全局吞吐（样本/s） | 每 rank batch | 预热更新 | 计时更新 |
| --- | --- | --- | --- | --- |
| 1p | 0.376 | 1 | 10 | 50 |
| 4p | 1.206 | 1 | 10 | 50 |

实测包含真实数据加载、传输、前向、反向及优化器更新，各更新同步；使用不同物理卡的单个逻辑芯片。主机 CPU/存储有其他任务共享，没有提供生产性能目标；本表不作为官方 FPS 或性能达标证明。

## 验证依据与复现

[逐项实机结果及原门槛](validation/extended_acceptance_20261018.json)记录实际执行阶段、退出码、输入范围、权重身份、数值/恢复/持续训练及未完成项。复现命令和结果文件见[完整训练与验证指导](FULL_VALIDATION.md)。旧十步基线保留在 `validation/train_result.json` 和 `validation/validation_result.json`，不能覆盖本节当前结果。

三项实际修复、CPU 参考范围、原失败及五份权重逐项结果见[数值修复说明](NUMERICAL_REPAIR.md)与 [446 参数清单](validation/selected_stability_gradient_parameters_20261018.json)。

恢复测试比较完整模型、优化器及适用的调度器、训练游标和各 rank RNG；单纯能加载模型不等于续训通过。CPU/NPU 使用相同输入与相同权重的数值对照，与数据集任务指标分别报告。

# 版本说明

## 变更

- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示／现象 | 处理 |
| --- | --- |
| 缺少镜像、固定权重或数据 | 完整解压本模型 ZIP，并先执行包内清单验证 |
| 旧 profile 的检查点不能精确续训 | 按完整指南选择同一三项数值 profile 的 schema2 检查点 |
| 验证要求明确 checkpoint | 使用完整指南列出的五个固定路径，每次选择新的输出目录 |
| 物理卡占用或逻辑 ID 重复 | 核对宿主 npu-smi 和物理映射，空闲后再运行 |
| 原12epoch或论文精度需求 | 按作者完整协议另行准备和验收，不以 mini 结果替代 |

## 许可

原作者 Apache-2.0 许可保留在 `vendor/ResWorld/LICENSE`；本项目新增脚本许可见本模型 LICENSE。
