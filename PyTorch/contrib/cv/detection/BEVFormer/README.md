# BEVFormer-base for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

BEVFormer 从多相机图像构建 BEV 表示。本工程迁移 BEVFormer-base，用于三维目标检测。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/fundamentalvision/BEVFormer.git
  commit_id=66b65f3a1f58caf0507cb2a971b9c0e7f842376c
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/README.md)。
- 本页章节参考：[Ascend DrivingSDK BEVFormer 原指导](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/BEVFormer/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

- [上游数据／项目说明](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/docs/prepare_dataset.md)；[DrivingSDK 迁移说明](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/BEVFormer/README.md)。

## 迁移说明

使用固定 DrivingSDK BEVFormer 补丁及本目录兼容补丁；将官方八卡样例调整为两卡 mini 功能配置，并提供独立镜像、数据检查和验证入口。

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
cd BEVFormer
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 准备数据集和预训练权重

```text
data/nuscenes/{samples,sweeps,maps,v1.0-mini}
data/annotations_original/nuscenes_infos_temporal_{train,val}.pkl
sources/BEVFormer/ckpts/r101_dcn_fcos3d_pretrain.pth
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

本次功能配置：**nuScenes mini 323 train / 81 val；FP16，2 卡每卡 batch 1，学习率 5e-5，1 epoch / 162 iter**。输出 `epoch_1.pth`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 作者已训练权重的完整 mini 评测

先完成本模型的 setup / prepare_data，在空闲单卡执行：

```bash
bash test/evaluate_mini_1p.sh --device_id=0
```

默认使用作者发布的 BEVFormer-base R101-DCN 24 epoch 权重，身份固定于 [evaluation.weights.lock.json](evaluation.weights.lock.json)。先核对 276,980,568 bytes 和 SHA256，再 strict 载入 1,089 个状态张量，按原顺序、原预处理计算全部 **81 个 mini_val 帧**的原仓 mAP/NDS。作者权重评测与本项目完整 trainval 训练分别验收。

仅有 GitHub 源码时，先补齐本模型镜像、固定源码、数据资产，再取得作者权重：

```bash
mkdir -p weights
curl --fail --location --retry 5 \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth \
  -o weights/bevformer_r101_dcn_24ep.pth
echo '5fda3ed3e0d2d1dc09baca337aa1ed64aece057fed7d8238525c5ed342649177  weights/bevformer_r101_dcn_24ep.pth' | sha256sum -c -
bash test/evaluate_mini_1p.sh --device_id=0
```

该 SHA256 是本项目对作者官方 HTTPS 发布文件的实际观测值，作者未单独发布 SHA256。10 月 18 日旧网盘 ZIP 不包含本次新增权重和入口，不能据新指导声称旧 ZIP 已更新。

历史 162 更新权重仅用于显式诊断，原失败精度记录保留：

```bash
bash test/evaluate_mini_1p.sh --device_id=0 --checkpoint-profile=delivered-mini
```

每次创建新的 `test/output/evaluate-mini_<时间>/`，包含原始终端日志、逐帧检查与指标 `mini_evaluation.json`、实际镜像/卡号/命令/退出码 `evaluation_execution.json` 和原始预测文件。已有训练指针和历史结果保留。可用 `--output_dir="$PWD/test/output/自定义新目录"` 指定新的输出目录。

## 从原 FCOS3D 初始化重新训练 mini

用于继续定位旧 162 更新检查点的低指标，采用全部 323 train / 81 val、两张物理卡、每卡 batch=1、FP16 / loss_scale=512，原 24 epoch 日程，共 3,888 更新；学习率沿用两卡 5e-5，保留作者时序 runner、网络、增强及固定配置的 500 更新 warmup / 余弦策略。只加载原 `r101_dcn_fcos3d_pretrain.pth`，不使用作者已训练 BEV 检测权重，也不覆盖旧短训结果。

```bash
bash test/train_mini_2p.sh --device_id=0,2
```

入口创建独立 `test/output/train-mini_<时间>/`，优化前核对原骨干张量；每步记录原始 head、损失、梯度有限性及各 rank 的 `quality_training_rank*.jsonl`。完成全部 24 轮后检查 `epoch_24.pth` 的模型/优化器、摘要与 rank 完成记录，自动重载并评测全部 81 帧。单独复测使用：

```bash
bash test/evaluate_mini_1p.sh --device_id=0 \
  --checkpoint="$PWD/test/output/<本次训练目录>/epoch_24.pth"
```

新入口首轮误覆盖原时序 runner，在首次更新前因 `eval_model` 参数失败；现保留原 runner，只修改轮数。实际启动、失败及逐 rank 更新见[实机进度](validation/mini_training_progress_20261020.json)。这仍是 mini 训练诊断，当前没有新检查点的最终指标；数据包含已声明的零 CAN 功能参考，不能代替正式 trainval 收敛或 GPU 对齐。

## 完整数据训练

完整 nuScenes 放在 `data/nuscenes_full/`（samples、sweeps、maps、v1.0-trainval），官方生成的两份 trainval PKL 放在这个目录；其 metadata.version 必须是 v1.0-trainval。 官方 CAN bus 放在 `data/can_bus/`，至少需相应 scene-*_pose.json，空目录会被拒绝。

```bash
bash test/train_full_2p.sh --device_id=0,2 --data_path="$PWD/data/nuscenes_full"
```

正式原始数据按以下结构放入本模型目录：

```text
data/nuscenes_full/{samples,sweeps,maps,v1.0-trainval}
data/can_bus/scene-*_pose.json
```

完整入口先核对 850 场景 / 34,149 个 sample。若两份 `nuscenes_infos_temporal_{train,val}.pkl` 均不存在，自动调用固定上游的 `create_nuscenes_infos(..., version='v1.0-trainval', max_sweeps=10)`；已有 PKL 时校验后使用，只有一份时拒绝覆盖。准备结果须有 28,130 train / 6,019 val，token 唯一、无交集、覆盖全部 sample，媒体文件存在且 CAN 向量不能全部为零。随后重建本模型完整数据路径。

转换入口通过 [cpu_data_converter.py](tools/cpu_data_converter.py) 隔离两项模型算子导入，从固定镜像内的原数据集源码提取 `NameMapping`，保留作者转换函数 AST；不修改坐标、标签、CAN、sweep 规则或训练网络。独立 CPU 对照可在 setup 后执行，下面命令不挂载 NPU：

```bash
docker run --rm --network none --cpus 4 --memory 16g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -v "${PWD}:${PWD}" -w "$PWD" \
  --entrypoint /opt/conda/envs/bevformer/bin/python \
  local/migration-drivingsdk:26.1.0-a3-headless-20261010 \
  test/check_cpu_data_converter.py \
  --output_dir "$PWD/test/output/cpu_converter_control_$(date +%Y%m%d_%H%M%S)"
```

2026-10-20 实跑退出 0：全部 mini 323 train / 81 val 的 17,980 个数组、123,038 个标量无差异；仅比较时归一化媒体路径前缀。该对照显式使用与已交付 mini 标注一致的零 CAN 参考输入，避免随后补齐正式 CAN 改变对照输入；正式转换仍必须使用真实官方 CAN。mini 作为完整数据输入时退出 1，准备目录、源码数据链接及结果指针保持。[命令、对照结果、数据/源码摘要与队列状态](validation/original_converter_control_20261020.json)。这验证转换算法保持，不代表完整 trainval 转换已通过。

正式训练队列已等待原始数据全覆盖核验，之后先执行完整转换，再等待空闲双物理卡启动上述公开完整入口。**当前完整转换和完整训练尚未完成，收敛与正式精度未验收。** `validate_1p.sh` 的判据对应既有功能配置；完整训练按上游官方评估说明另行验收。

# 训练结果展示

## 本项目实测（2026-10-12）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），不是官方训练系列。以下为本仓独立入口的实际结果。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | nuScenes mini：323 train / 81 val；CAN 扩展缺失的 404 条记录使用零填充 |
| 训练配置 | 2 张物理卡；FP16 混合精度（loss_scale=512）；全局 batch=2；1 epoch / 162 次更新 |
| 保存状态 | 1,089 个模型张量、557 组优化器状态有限；优化器 step=162 |
| 独立验证 | strict 重载；1 帧真实输入推理输出有限 |
| 未完成 | 完整 nuScenes mAP/NDS、收敛、吞吐、续训及长稳；单帧有限性不等于检测精度 |

当前检查点 SHA256：`51af21f51b5696fba4336055598a6cd2ea9cdc1fae97c9879fcfad06b8c49458`。

执行时间（UTC）：`2026-10-12T09:22:45.452936+00:00` 至 `2026-10-12T09:30:26.027942+00:00`。原始执行命令：

```bash
bash test/train_2p.sh --device_id=0,2
bash test/validate_1p.sh --device_id=0
```

两条命令退出码均为 0。宿主卡号是当次资源分配记录；复现时按本页环境与卡号映射选择空闲设备。

证据：[执行与退出码](validation/entry_acceptance.json)、[训练摘要](validation/reference_run/train_result.json)、[模型/优化器检查](validation/reference_run/saved_state_check.json)、[独立验证数据](validation/reference_run/validation_result.json)。

另已完成[单镜像导入](validation/single_image_import.json)、[更换模型目录复验](validation/path_relocation.json)及[完整数据负向门禁](validation/full_data_guard.json)。功能数据不能作为完整训练数据通过门禁。

## 2026-10-20 全部 mini 验证帧与任务指标

后续作者权重对照：全部 81 帧有限，**mAP=0.3623209477、NDS=0.4103581189**。显式 `--checkpoint-profile=official-r101-24ep` 与默认公开入口分别实跑至退出 0，原始指标相同。同一输入与评测器下，权重身份解释了此前短训检查点接近零的结果；未因此宣称本项目完整训练或 GPU/NPU 精度对齐已验收。[两次执行、原始指标、逐帧结果与权重身份](validation/official_checkpoint_mini_evaluation_20261020.json)。

全部 81 个标准 mini_val 帧的预测有限，原仓检测评测器 mAP=0.00003540314127、NDS=0.0178060191423（均为 0～1 比例）；公开入口退出 0。检查点来自既有 162 更新短训，任务质量未验收。

[逐帧 token、原始指标、检查点 SHA、镜像、实际命令及退出码](validation/complete_mini_evaluation_20261020.json)。实测来自同一固定镜像的新目录，单卡与 Qwen 双卡训练并行；计时属于并行功能验证，不作为隔离性能基线。当前完整 nuScenes trainval 训练、收敛和正式任务精度仍未完成。

## 通过判据

训练必须退出 0，并产生 `pass_functional_training`；独立验证必须退出 0，并产生 `pass_independent_functional_validation`。保存的模型/优化器、实际更新次数及上述推理范围均须检查通过。正式任务精度尚未测量的项目不填入估计值。

# 版本说明

## 变更

- 2026-10-20：默认完整 mini 评测改用有来源、大小和摘要记录的作者已训练权重，两次 81 帧指标相同；旧短训检查点另设 profile 保留。新增原转换器 CPU 入口及全字段对照、完整数据转换准备和训练等待队列；正式任务精度继续独立验收。
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
