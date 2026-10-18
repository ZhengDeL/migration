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

## 当前实测

- 原156参数失败及后续1参数失败均已修复；五份固定权重各446个有效梯度全部通过原阈值。
- 第三项修复前后446个CPU参考梯度逐位相同；此前越界参数差由0.0025835降至0.0000926。
- 四卡10+10对20：2054/2054完整状态张量逐位相同，四rank实际增强输入一致。
- 四卡1811.968秒持续训练、最终权重数值与独立重载、作者全部65帧原规划/碰撞对照通过。

三处实际修改为官方erf GELU、等价稳定时序残差、固定常量几何投影CPU FP32路径。NN特征、采样和反向仍在NPU。根因、历史失败与CPU参考范围见[数值修复说明](NUMERICAL_REPAIR.md)。正式nuScenes trainval/12epoch及驾驶业务精度尚未验收。

## 通过判据

2026-10-18在7.6.28.252的固定A3环境完成本次数值修复及独立回归。当前结果为 `pass_mini_numerical_and_operational_gates_full_protocol_pending`；逐项实测见[JSON](validation/extended_acceptance_20261018.json)。

| 检查 | 实际结果 |
| --- | --- |
| 原bf7a失败、后续07a6失败、旧72afe、新训练20步、最终长稳五份权重 | 每份446有效梯度全部通过原逐参数、整体L2、loss和一步参数门槛 |
| 常量几何修改前后CPU参考 | 446/446完整梯度逐位相同 |
| 四卡10+10恢复 vs连续20 | 2054/2054状态张量逐位相同、max0；全部rank增强后真实输入一致 |
| 四卡持续训练 | 1811.967955秒、660更新；模型/优化器有限、参数更新、rank参数SHA一致 |
| 最终权重独立梯度 / 重载 | pass；relativeL2=2.83294570892e-05；独立重载通过 |
| 作者完整权重原65帧 | CPU/NPU轨迹和逐帧原离散碰撞对照通过 |
| 正式trainval/12epoch和驾驶业务精度 | 尚未完成 |

原逐参数／整体L2／loss／一步更新门槛、完整参数清单和各权重身份见 [数值验证章节](FULL_VALIDATION.md#4-数值修复与五份固定权重回归)。独立包还完成了整包 SHA256 核对、新目录实际解压及包内入口重载复验。

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
