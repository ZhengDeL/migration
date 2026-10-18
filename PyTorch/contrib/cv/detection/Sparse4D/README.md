# Sparse4Dv3 for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

Sparse4D 通过稀疏关键点采样与时序特征融合进行三维目标检测。本工程采用 Sparse4Dv3。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/HorizonRobotics/Sparse4D.git
  commit_id=249ffbb695f4e9db628d953e2bf6d36de04bbb69
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/docs/quick_start.md)。
- 本页章节参考：[Ascend DrivingSDK Sparse4D 原指导](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/Sparse4D/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

- [上游数据／项目说明](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/docs/quick_start.md)；[DrivingSDK 迁移说明](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/Sparse4D/README.md)。

## 迁移说明

使用固定 DrivingSDK Sparse4D 补丁及辅助文件；同步调整两卡全局 batch 与学习率，功能训练使用普通 AdamW；不把官方八卡结果记作本项目结果。

# 准备训练环境

## 安装昇腾环境

- 服务器：ARM64 Linux、配套 Ascend 驱动、Docker、Python 3.9+，训练需要 2 张空闲物理卡；模型 Python 与运行库在镜像内。`+cpu` 包版本标记不表示使用 CPU 训练，实际设备由环境检查和实测报告确认。
- 本次 1# 为 Atlas 800I A3 推理系列，实验训练结果不改变华为官方训练硬件支持范围。

宿主驱动与容器的 CANN/TorchNPU 配套需匹配。每张物理卡只取一个宿主逻辑 ID，具体拓扑先用 `npu-smi info` 核对。

## 安装模型环境

| 项目 | 本项目固定版本 |
| --- | --- |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| image ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| CANN | `9.1.0` |
| Python | `3.10.21` |
| PyTorch | `2.7.1+cpu` |
| TorchNPU | `2.7.1.post6` |
| DrivingSDK | `26.1.0.post2` |

取得本模型完整离线包并解压，进入包含本 README 的目录。包内镜像为一个独立 Docker 归档；原始数据、源码和权重均属于本模型，数据准备不访问网络。

```bash
cd Sparse4D
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 准备数据集和预训练权重

```text
data/nuscenes/{samples,sweeps,maps,v1.0-mini}
data/annotations_original/nuscenes_infos_{train,val}.pkl
sources/Sparse4D/nuscenes_kmeans900.npy
sources/Sparse4D/ckpt/resnet50-19c8e357{,-modern}.pth
```

`sources.lock.json` 固定本模型源码和补丁。本模型使用本目录保存的DrivingSDK固定补丁/辅助文件，以及镜像内mx_driving。 迁移到新目录后重新运行本模型的 `bash test/prepare_data.sh`。

仅源码部署时，按上述原仓数据说明补齐资产；在本模型固定镜像中运行 `tools/fetch_sources.py --execute`、`tools/prepare.py`，再遵循上述官方数据说明准备资产。离线交付已经含固定源码，不需重新下载或重新安装依赖。

# 开始训练

## 训练模型与独立验证

```bash
npu-smi info
bash test/check_env.sh --device_id=0,2
bash test/plan.sh
bash test/train_2p.sh --device_id=0,2
bash test/validate_1p.sh --device_id=0
```

1# 一张物理卡有两个逻辑芯片；默认 `0,2` 每物理卡只取一个。容器内重新编号为连续 ID。换机先核对实际拓扑，换卡修改 `--device_id`。验证只用一张卡，只填一个 ID。

每次训练创建独立的 `test/output/functional_<时间>/`，保留已有结果；`test/output/latest_run.json` 指向最近一次尝试。训练失败时验证脚本拒绝把旧结果当成本次通过。所有原始终端输出、配置、checkpoint 和 JSON 报告均保存于本模型目录内。

本次功能配置：**nuScenes mini；2 卡每卡 batch 1，普通 AdamW，学习率 2.5e-5，10 次优化迭代**。输出 `iter_10.pth`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 完整数据训练

完整 nuScenes 放在 `data/nuscenes_full/`（samples、sweeps、maps、v1.0-trainval），官方生成的两份 trainval PKL 放在这个目录；其 metadata.version 必须是 v1.0-trainval。 按上游 kmeans 工具在完整训练标注上生成 `data/nuscenes_full/nuscenes_kmeans900.npy`；完整入口使用它，不复用 mini anchor。

```bash
bash test/train_full_2p.sh --device_id=0,2 --data_path="$PWD/data/nuscenes_full"
```

完整入口检查数据版本并重新建立完整数据路径，缺少正式资产会退出。完整nuScenes需放入本模型数据目录，并按官方说明生成完整标注和CAN bus；mini标注不能复用为完整数据验收。 **本次没有执行完整数据训练，未验证完整训练收敛或精度。** `validate_1p.sh` 的当前判据对应本次功能配置，不能直接当成完整训练评估脚本；完整训练按上游官方评估说明验收。

# 训练结果展示

## 本项目实测（2026-10-12）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），不是官方训练系列。以下为本仓独立入口的实际结果。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | nuScenes mini 功能子集，非完整 trainval |
| 训练配置 | 2 张物理卡；FP16 混合精度（loss_scale=32）；全局 batch=2；10 次更新 |
| 保存状态 | 819 个模型张量、652 组优化器状态有限；优化器 step=10 |
| 独立验证 | strict 重载；1 帧真实输入推理输出有限 |
| 未完成 | 完整 nuScenes 检测指标、收敛、吞吐、续训及长稳；十步不能证明收敛 |

当前检查点 SHA256：`f948ad2a2d5f6eb8769e488d3cbad07771bbd23251ae939144d9d499ec3d6f66`。

执行时间（UTC）：`2026-10-12T09:22:45.457686+00:00` 至 `2026-10-12T09:24:16.757376+00:00`。原始执行命令：

```bash
bash test/train_2p.sh --device_id=4,6
bash test/validate_1p.sh --device_id=4
```

两条命令退出码均为 0。宿主卡号是当次资源分配记录；复现时按本页环境与卡号映射选择空闲设备。

证据：[执行与退出码](validation/entry_acceptance.json)、[训练摘要](validation/reference_run/train_result.json)、[模型/优化器检查](validation/reference_run/saved_state_check.json)、[独立验证数据](validation/reference_run/validation_result.json)。

另已完成[单镜像导入](validation/single_image_import.json)、[更换模型目录复验](validation/path_relocation.json)及[完整数据负向门禁](validation/full_data_guard.json)。功能数据不能作为完整训练数据通过门禁。

## 通过判据

训练必须退出 0，并产生 `pass_functional_training`；独立验证必须退出 0，并产生 `pass_independent_functional_validation`。保存的模型/优化器、实际更新次数及上述推理范围均须检查通过。正式任务精度尚未测量的项目不填入估计值。

# 版本说明

## 变更

- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示 / 现象 | 处理 |
| --- | --- |
| 缺少镜像归档 / image ID 不一致 | 取得本模型完整包，核对 images.lock.json 与现有镜像用途 |
| 标注或图像路径缺失 | 重新运行本模型 prepare_data，核对 data 目录；不要生成假标签 |
| NPU 不空闲 / 卡号重复 | 核对宿主进程和物理映射，处理占用后再启动 |
| 只有训练日志没有通过报告 | 查看本次输出最后异常与退出码，不能据此宣称通过 |
| 迁入另一目录后文件路径错误 | 先 setup/prepare_data，再检查 plan，使用当前目录的新配置 |

本模型实机证据随验证报告提供；问题档案和内部操作记录留在本机，不随离线包上传。
