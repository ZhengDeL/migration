# Apollo VisionNet for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

本工程迁移 Apollo VisionNet 的 BEV/OCC 模型训练路径，沿用官方 DLA 深度初始化与原始 OCC 标注。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/ApolloAuto/Apollo-Vision-Net.git
  commit_id=4909aa681bc435a74e793e714b8066d9bbf0095d
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/run.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

- [上游数据／项目说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/prepare_dataset.md)；[上游训练说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/run.md)。

## 迁移说明

以本目录 ascend.patch / compat.patch 处理 NPU 和依赖兼容，编译 FocalLoss 改为已有 PyTorch 公式，并保留独立前后向对照证据。

# 准备训练环境

## 安装昇腾环境

- 服务器：ARM64 Linux、配套 Ascend 驱动、Docker、Python 3.9+，训练需要 4 张空闲物理卡；模型 Python 与运行库在镜像内。`+cpu` 包版本标记不表示使用 CPU 训练，实际设备由环境检查和实测报告确认。
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
cd Apollo-Vision-Net
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 准备数据集和预训练权重

```text
data/nuscenes/{samples,sweeps,maps,v1.0-mini}
data/apollo/nuscenes_infos_temporal_{train,val}_occ_gt.pkl
data/apollo/occ_gt_release_v1_0.tar.gz 或已展开目录
sources/Apollo-Vision-Net/ckpts/官方DLA深度预训练权重.pth
```

`sources.lock.json` 固定本模型源码和补丁。本模型的NPU与兼容补丁保存在本目录，来源和SHA见sources.lock.json。 迁移到新目录后重新运行本模型的 `bash test/prepare_data.sh`。

仅源码部署时，按上述原仓数据说明补齐资产；在本模型固定镜像中运行 `tools/fetch_sources.py --execute`、`tools/prepare.py`，再遵循上述官方数据说明准备资产。离线交付已经含固定源码，不需重新下载或重新安装依赖。

# 开始训练

## 训练模型与独立验证

```bash
npu-smi info
bash test/check_env.sh --device_id=0,2,4,6
bash test/plan.sh
bash test/train_4p.sh --device_id=0,2,4,6
bash test/validate_1p.sh --device_id=0
```

1# 一张物理卡有两个逻辑芯片；默认 `0,2,4,6` 每物理卡只取一个。容器内重新编号为连续 ID。换机先核对实际拓扑，换卡修改 `--device_id`。验证只用一张卡，只填一个 ID。

每次训练创建独立的 `test/output/functional_<时间>/`，保留已有结果；`test/output/latest_run.json` 指向最近一次尝试。训练失败时验证脚本拒绝把旧结果当成本次通过。所有原始终端输出、配置、checkpoint 和 JSON 报告均保存于本模型目录内。

本次功能配置：**官方 DLA 222 张量初始化；242 train / 162 独立 val；4 卡 FP32 每卡 batch1，学习率1e-4，10 步**。输出 `iter_10.pth`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 交付 mini 检查点的完整验证集评测

先完成本模型的 setup / prepare_data，在空闲单卡执行：

```bash
bash test/evaluate_mini_1p.sh --device_id=0
```

入口固定使用本模型 `validation/reference_run/iter_10.pth`，先核对交付 SHA256，再 strict 重载模型。按原数据划分、顺序和预处理运行全部 **162 个 mini 验证帧**，计算原仓 `evaluate_occ_iou` / `SSCMetrics` 的 OCC IoU、mIoU、各类 IoU、precision、recall。本轮使用已有短训检查点，完整 trainval 训练和收敛精度仍须按完整数据协议验收。

每次创建新的 `test/output/evaluate-mini_<时间>/`，包含原始终端日志、逐帧检查与指标 `mini_evaluation.json`、实际镜像/卡号/命令/退出码 `evaluation_execution.json` 和原始预测文件。已有训练指针和历史结果保留。可用 `--output_dir="$PWD/test/output/自定义新目录"` 指定新的输出目录。

Apollo mini OCC 的独立验证集为 162 帧；原仓 nuScenes detection 入口要求标准 mini_val 的 81 帧。本入口按原 OCC 划分计算占用指标，检测 mAP/NDS 留待对应正式数据协议；不修改原评测器或替换划分。IoU/mIoU 使用百分数，precision/recall 使用 0～1 比例。

## 从官方骨干初始化重新训练 mini

用于定位十步权重 mIoU=0：全部 242 个训练帧，四张物理卡、FP32、每卡 batch=1、24 epoch / 1,464 更新；保留原图像增强、网络、损失、24 epoch 余弦策略及 500 更新 warmup，学习率按原八卡 2e-4 缩放为四卡 1e-4。初始仅载入官方 DLA 深度骨干，不载入完整作者检测/OCC 模型。

```bash
bash test/train_mini_4p.sh --device_id=0,2,4,6
```

入口创建新的 `test/output/train-mini_<时间>/`；每一步检查原始 head、损失和梯度有限性，并写入各 rank 的 `quality_training_rank*.jsonl`。完成训练后检查最终模型/优化器，自动重载 `epoch_24.pth`，计算全部原 OCC 162 个独立留出帧指标。结果分别保存在 `quality_training_rank*.json`、`mini_evaluation.json`，不覆盖旧十步检查点。

新发现并修复原数据对象的多进程兼容问题：nuScenes 配置的 `class_names` 为不可 pickle 的 `dict_keys`，在本模型构造时转为保持原顺序的列表；保留每 rank 4 个 DataLoader worker。完整训练、自动重载评测与失败尝试见 [实机记录](validation/mini_training_evaluation_20261020.json)。本流程仍为 mini 训练诊断，正式 trainval 训练独立执行。

训练完成后如需重新评测，在新输出目录执行：

```bash
bash test/evaluate_mini_1p.sh --device_id=0 \
  --checkpoint="$PWD/test/output/<本次训练目录>/epoch_24.pth"
```

## 完整数据训练

完整原始 nuScenes 放在 `data/nuscenes_full/`，完整 OCC 展开在 `data/apollo/occ_gt_release_v1_0/`；官方两份 OCC train/val PKL 放在 `data/apollo/`。 官方 CAN bus 放在 `data/can_bus/`，至少需相应 scene-*_pose.json，空目录会被拒绝。

```bash
bash test/train_full_4p.sh --device_id=0,2,4,6 --data_path="$PWD/data/apollo"
```

完整入口检查数据版本并重新建立完整数据路径，缺少正式资产会退出。完整nuScenes需放入本模型数据目录，并按官方说明生成完整标注和CAN bus；mini标注不能复用为完整数据验收。 **本次没有执行完整数据训练，未验证完整训练收敛或精度。** `validate_1p.sh` 的当前判据对应本次功能配置，不能直接当成完整训练评估脚本；完整训练按上游官方评估说明验收。

# 训练结果展示

## 本项目实测（2026-10-12）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），不是官方训练系列。以下为本仓独立入口的实际结果。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | 官方划分与 nuScenes mini 交集：242 train / 162 val，训练验证不相交；完整 OCC 归档结构已校验 |
| 训练配置 | 官方 DLA 深度预训练初始化；4 张物理卡；FP32；10 次更新 |
| 保存状态 | 600 个模型张量、465 组优化器状态有限；优化器 step=10 |
| 独立验证 | strict 重载；全部 162 个独立验证帧的原始 occupancy head 和解码结果有限 |
| 未完成 | 完整 nuScenes 检测 mAP / 占用 mIoU、收敛、吞吐、续训及长稳；有限值检查不等于任务精度 |

当前检查点 SHA256：`75f808d0d52b208b804508af048f39c22c10ffcbb14a28c7b299afc879c7a9c3`。

执行时间（UTC）：`2026-10-12T09:36:21.917763+00:00` 至 `2026-10-12T09:39:29.695811+00:00`。原始执行命令：

```bash
bash test/train_4p.sh --device_id=8,10,12,14
bash test/validate_1p.sh --device_id=8
```

两条命令退出码均为 0。宿主卡号是当次资源分配记录；复现时按本页环境与卡号映射选择空闲设备。

证据：[执行与退出码](validation/entry_acceptance.json)、[训练摘要](validation/reference_run/train_result.json)、[模型/优化器检查](validation/reference_run/saved_state_check.json)、[独立验证数据](validation/reference_run/validation_result.json)。

另已完成[单镜像导入](validation/single_image_import.json)、[更换模型目录复验](validation/path_relocation.json)及[完整数据负向门禁](validation/full_data_guard.json)。功能数据不能作为完整训练数据通过门禁。

## 2026-10-20 全部 mini 验证帧与任务指标

全部 162 个原 OCC heldout 帧预测有限；原仓 OCC IoU=0%、mIoU=0%、precision=0、recall=0；公开入口退出 0。检查点来自既有 10 更新短训，任务质量未验收。修复 `prepare_apollo_mini.py` 遗漏的 `occ_invalid_path`：404 份原始可见性掩码逐份校验、复制并重建路径，原标签和 mask 内容、0.25 阈值及评测公式保持原值。另对保存的原始预测做只读 CPU 检查：162 帧在原 0.25 阈值下均无占用体素输出，这解释了零 IoU；没有降低阈值来制造通过。

[逐帧 token、原始指标、检查点 SHA、镜像、实际命令及退出码](validation/complete_mini_evaluation_20261020.json)。实测来自同一固定镜像的新目录，单卡与 Qwen 双卡训练并行；计时属于并行功能验证，不作为隔离性能基线。当前完整 nuScenes trainval 训练、收敛和正式任务精度仍未完成。

## 2026-10-20 新 mini 训练最终结果

仅从官方 DLA 深度骨干初始化，四卡各完成 **1,464 更新 / 24 epoch**；每 rank 的 222 个初始化张量通过核验，原始输出、损失、465 个有效梯度/步及保存状态有限。末尾 strict 重载 `epoch_24.pth`，全部 **162/162** 原 OCC 留出帧完成原 `SSCMetrics` 评测，退出 0：**IoU=18.6466263092%、mIoU=4.7205596444%**。

新权重已产生非零 OCC 指标；原标签、可见性 mask、0.25 阈值和评测公式未为提高分数而更改。旧十步权重 mIoU=0 的记录仍保留。此 mini 结果仍不足以验收正式质量；完整 trainval 训练、检测/OCC 指标及 GPU 对齐待完成。[实际命令、四 rank 完成记录、完整混淆矩阵/逐帧检查和检查点 SHA](validation/mini_training_evaluation_20261020.json)。

## 通过判据

原功能短训 `train_4p.sh` / `validate_1p.sh` 分别对应退出 0、`pass_functional_training` / `pass_independent_functional_validation`。

新增 `train_mini_4p.sh` 的流程判据是：四 rank 的 `quality_training_rank*.json` 均为 `pass_completed_mini_training`、各 1,464 更新；自动完整 162 帧 OCC 评测退出 0，`mini_evaluation.json` 为 `pass_complete_mini_occupancy_evaluation`，模型、优化器及预测有限。独立复测使用本页 `evaluate_mini_1p.sh --checkpoint=...`。以上是 mini 流程/数值判据，正式 trainval、检测/OCC 任务质量单独验收。

FocalLoss 兼容路径的 8 个有限 CPU/NPU 前后向案例通过 `atol=1e-5, rtol=1e-4`；[算子数值记录](validation/focal_parity_20261011.json)保留实际误差，不能代替整模型检测/占用指标。

## 2026-10-19 完整 OCC 数据准备

完整 `occ_gt_release_v1_0.tar.gz` 已校验两端 SHA256 并解压。700 train 场景 / 28,130 样本、150 val 场景 / 6,019 样本，全部 102,447 个 NPY 标签文件的 header 与数据长度通过检查；与正式 nuScenes 场景/样本完全对应，训练验证无重叠。见[完整检查数据](validation/full_occ_data_20261019.json)。相机/点云、CAN、派生标注与完整训练/评测仍在准备，不能以标签完整代替模型 mAP/mIoU。

# 版本说明

## 变更

- 2026-10-20：新增完整 mini OCC 原仓评测；修复原始可见性 mask 的复制/路径/摘要/引用集合校验，公开 162 帧 IoU/mIoU=0 的实际结果。
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
