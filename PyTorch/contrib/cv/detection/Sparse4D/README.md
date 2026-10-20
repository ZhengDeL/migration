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

## 全部 mini 验证帧的独立检测评测

默认使用作者发布的 **Sparse4Dv3 R50 已训练权重**，文件身份固定在 [evaluation.weights.lock.json](evaluation.weights.lock.json)。先完成本模型 setup / prepare_data，再将权重放入本模型目录：

```bash
mkdir -p weights
wget -c -O weights/sparse4dv3_r50.pth \
  https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth
bash test/evaluate_mini_1p.sh --device_id=0
```

离线机器将同一文件复制到 `weights/sparse4dv3_r50.pth`；入口先按 lock 的 SHA256 核验，再 strict 重载 819 个状态张量，以原配置运行全部 **81 个标准 mini_val 帧**，计算原仓 detection mAP/NDS 及各类别指标。默认权重约 577 MB，SHA256 为 `5beed4d4933ca6448d72586b0f8812863574289ff3c4192de71dc9f46a42f0ed`；这是本项目从官方 HTTPS 资产观察并固定的摘要，发布方未另提供 SHA256。

每次创建新的 `test/output/evaluate-mini_<时间>/`，包含 `console_evaluate-mini.log`、逐帧检查和指标 `mini_evaluation.json`、镜像/卡号/命令/退出码 `evaluation_execution.json`、原始预测和 nuScenes 指标文件。已有训练指针和历史结果保留；可用 `--output_dir="$PWD/test/output/自定义新目录"` 指定新的输出路径。本入口只评测检测，完整 trainval 精度与本项目从头训练仍另行验收。

旧十步交付检查点仅供失败复现：`bash test/evaluate_mini_1p.sh --device_id=0 --checkpoint-profile=delivered-mini`。该身份已知在第 36 帧失败，不能用于任务质量验收。当前网盘旧 ZIP 尚不包含新脚本/作者权重；新完整离线包将在本轮完整验收后按模型独立归档。

## 从 ImageNet 骨干重新训练 mini

旧十步权重溢出的进一步训练验证使用全部 323 个 mini 训练帧和独立 81 个 mini_val 帧。从原 ImageNet ResNet50 骨干初始化，不使用作者完整检测权重，也不继续训练已失败的旧十步状态。

```bash
bash test/train_mini_2p.sh --device_id=0,2
```

保留原配置的每卡 batch=6，两卡全局 batch=12，学习率从原 batch=48 的 6e-4 线性缩放为 1.5e-4；保留 FP16/loss_scale=32、原网络/增强/时序 sampler、100 个按迭代定义的 mini epoch 和 500 更新 warmup。mini 的每轮按原公式取 `323 // 12 = 26`，共 2,600 更新；这不是完整 trainval 的 100 epoch。

每次创建独立输出目录，检查原始 head/keypoint 输出、损失、梯度及最终模型/优化器，保留各 rank 逐步记录。训练完成后自动 strict 重载 `iter_2600.pth`，运行全部 81 个 mini_val 帧及原检测指标；不裁剪尺寸、不跳帧、不改指标。[带时间的实际训练状态](validation/mini_training_progress_20261020.json)记录已执行部分；结束前不能据启动或有限 loss 判定溢出风险及精度已经解决。

完成后可在新的评测目录单独重载：

```bash
bash test/evaluate_mini_1p.sh --device_id=0 \
  --checkpoint="$PWD/test/output/<本次训练目录>/iter_2600.pth"
```

## 完整数据训练

先按[固定版本上游的数据准备说明](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/docs/quick_start.md)准备完整 nuScenes，将 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/` 放在本模型的 `data/nuscenes_full/`。完整入口自动执行固定原仓 `tools/nuscenes_converter.py`，再以**完整训练集**执行原仓 `tools/anchor_generator.py`；不需要先启动模型，不复用 mini 的 PKL 或 anchors。

```bash
bash test/train_full_2p.sh --device_id=0,2 --data_path="$PWD/data/nuscenes_full"
```

入口按以下顺序准备与核验：

1. 核对原始元数据的 850 场景、34,149 样本；转换后逐个核对官方 train / val token 归属，必须为 28,130 / 6,019 帧且不重叠。
2. 仅用训练标注生成 900×11 anchors，记录训练 PKL、原工具和 anchors 的 SHA256、种子与参数到 `data/nuscenes_full/full_anchor_provenance.json`。已有但缺少来源收据的 anchors、仅一份 PKL、改名 mini 均拒绝继续，保留现场供核查。
3. 将核验后的标注与路径准备到 `prepared/full_nuscenes/`。两卡各 batch=6，全局 batch=12，lr=1.5e-4；保留原 100 轮、FP16/loss_scale=32 和余弦日程，计划 234,400 次更新。直接调用 `tools/run_train.py` 也必须通过完整准备收据及 anchors 身份检查。

可先在无 NPU 的 CPU 容器复核原转换器、原 anchors 生成、完整配置与 mini 冒充完整数据的拦截；该控制需要本模型已准备好的 mini 和原源码：

```bash
bash test/check_preparation_cpu.sh
```

独立输出目录中的 `full_preparation_control.json` 保存逐字段对照和负向检查，`preparation_execution.json` 保存实际命令、镜像与退出码。原 K-means 的并行浮点运算可能影响 anchors 字节摘要；以本次生成收据核验身份，不承诺跨环境逐位相同。

[2026-10-20 实跑证据](validation/full_preparation_control_20261020.json)：原转换器的全部 404 mini 帧，17,980 个数组、121,422 个标量对照零差异；原训练集 K-means、完整配置及两类错误入口拦截通过。**这不是全量正向转换或完整训练结果；正式 raw 数据尚未到齐，完整训练、收敛与精度未验收。** `validate_1p.sh` 对应历史功能配置；完整精度须使用完整 val 和上游原指标评测。

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

## 2026-10-20 完整 mini 评测发现的失败

固定十步交付检查点在第 36 帧（索引 35，token `a771effa2a2648d78096c3e92b95b129`）出现非有限预测；完整 81 帧评测**未通过**。固定种子复跑 FP16、FP32 均复现。FP32 解码器输入的对数尺寸为 89.4342651367，超过 FP32 `exp` 有限上限约 88.7228393555；将同一输入移到 CPU，`exp` 同样溢出。FP16 路径还捕获了 keypoint 生成中的首个非有限输出。

这份未收敛检查点不能用于完整推理或任务质量验收；此前十步训练和单帧有限性结果仅保留其原有范围。未截断尺寸、跳帧、屏蔽 NaN 或填写 mAP/NDS。[失败帧、原始模块诊断、FP16/FP32 和 CPU 控制、命令/退出码及权重 SHA](validation/complete_mini_failure_20261020.json)。作者发布的已训练权重已在同一固定镜像完成 81/81 帧对照，mAP=0.4635453126、NDS=0.5071118864；该控制说明旧十步权重不可代替已训练模型。正式 trainval 训练和完整指标仍待执行。

## 作者已训练权重对照（2026-10-20）

同一固定镜像与标准 mini_val 81 帧，strict 重载作者 Sparse4Dv3 R50 权重，全部预测有限；原仓评测 **mAP=0.4635453126、NDS=0.5071118864**（0～1 比例）。显式 `--checkpoint-profile=official-v3-r50` 与不加该参数的默认入口分别在两个新输出目录实跑，均退出 0，主要指标一致。

[全部帧、两个公开命令/退出码、镜像、权重/代码摘要及原始指标](validation/official_checkpoint_mini_evaluation_20261020.json)。这是作者已训练权重在 mini_val 的 NPU 检测控制；本项目完整 trainval 训练、全量精度及同条件 GPU/NPU 对照仍需完成。旧十步权重失败记录保留，不能将该失败权重替换身份后声称通过。

## 2026-10-20 新 mini 检查点的完整训练与复测

从原 ImageNet 骨干重新训练已完成：两 rank 各 **2,600 更新**，每卡 batch=6，模型及优化器保存状态有限。新检查点自动重载完成全部 **81/81** 留出帧；另通过公开 `test/evaluate_mini_1p.sh --checkpoint=.../iter_2600.pth` 在新输出目录复测，两次均退出 0，**mAP=0.0604165495、NDS=0.1097741302** 相同。

此次新检查点未复现旧十步权重的整套推理溢出；旧权重及失败证据仍保留。这是原 mini 数据上的完整声明训练流程，正式 trainval 的完整训练、收敛、任务质量和 GPU 对齐仍未验收，不能与作者全量训练权重的指标作同条件对照。[实际训练命令、各 rank 完成记录、检查点/源码 SHA 和两次全部指标](validation/mini_training_evaluation_20261020.json)。

## 通过判据

原功能短训 `train_2p.sh` / `validate_1p.sh` 分别对应退出 0、`pass_functional_training` / `pass_independent_functional_validation`。

新增 `train_mini_2p.sh` 的流程判据是：两 rank 的 `quality_training_rank*.json` 均为 `pass_completed_mini_training`、各 2,600 更新；自动完整 81 帧评测退出 0，`mini_evaluation.json` 为 `pass_complete_mini_detection_evaluation`，模型、优化器及预测有限。独立复测使用本页 `evaluate_mini_1p.sh --checkpoint=...`。本次流程已通过，实测 mAP/NDS 见前述终态 JSON；较低的 mini 指标不能作为正式 trainval 收敛/任务质量达标依据。

# 版本说明

## 变更

- 2026-10-20：新增完整 mini 评测和十步检查点第 36 帧真实失败/CPU exp 控制；作者权重对照按独立身份记录。
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
