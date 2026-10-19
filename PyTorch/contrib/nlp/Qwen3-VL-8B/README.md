# Qwen3-VL-8B-Instruct for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

本工程使用 Qwen3-VL-8B-Instruct，按 LLaMA-Factory 的多模态 SFT 流程完成 Ascend 两卡 LoRA 功能训练。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/hiyouga/LLaMA-Factory.git
  commit_id=1fded918ac4723ac9966363f51ed4500866c5d1d
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/hiyouga/LLaMA-Factory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/examples/README.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

- [上游数据／项目说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/data/README_zh.md)；[上游训练说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)。

## 迁移说明

沿用固定 LLaMA-Factory 的 NPU 多模态 SFT 与 LoRA 实现，以本目录两卡 BF16 入口管理演示数据、保存和独立图文重载。完整 DriveLM 任务新增以下适配：

- 将官方原始问答转换为六相机 ShareGPT 数据，按场景分开训练与本地留出数据。
- 用原生编码器检查全部问答长度，正式配置使用 4,096 token。
- 完整训练通过 PyTorch `torchrun` 调用固定上游 `run_exp()`；网络、像素 collator、LoRA 和优化器均使用该上游实现。
- [元数据复用](tools/drivelm_metadata_cache.py)只保留一个帧的原生返回值，按六个文件的路径、大小、修改时间和 processor 身份匹配；多条问答共用一帧时避免重复图片预处理。训练 collator 继续调用原生 `_get_mm_inputs` 生成当前 batch 像素。

完整原生预处理最初对每条问答重复处理六张图，实测仅每秒几十条；本次使用单帧复用处理该开销。改动前后以相同真实样本、适配器摘要、重载 loss 和完整预处理进度留证，见本页本轮实测记录。

# 准备训练环境

## 安装昇腾环境

- 服务器：ARM64 Linux、配套 Ascend 驱动、Docker、Python 3.9+，训练需要 2 张空闲物理卡；模型 Python 与运行库在镜像内。`+cpu` 包版本标记不表示使用 CPU 训练，实际设备由环境检查和实测报告确认。
- 本次 1# 为 Atlas 800I A3 推理系列，实验训练结果不改变华为官方训练硬件支持范围。

宿主驱动与容器的 CANN/TorchNPU 配套需匹配。每张物理卡只取一个宿主逻辑 ID，具体拓扑先用 `npu-smi info` 核对。

## 安装模型环境

| 项目 | 本项目固定版本 |
| --- | --- |
| 镜像 tag | `local/llamafactory:a3-arm64-20261012` |
| image ID | `sha256:99d33965ae4502b07bffe2c3248887795ae0672a0e79dd0aaa567d09cce811b5` |
| CANN | `9.1.0` |
| Python | `3.12.13` |
| PyTorch | `2.10.0+cpu` |
| TorchNPU | `2.10.0.post2` |
| transformers | `5.8.0` |
| peft | `0.18.1` |
| llamafactory | `0.9.6.dev0` |
| accelerate | `1.11.0` |

取得本模型完整离线包并解压，进入包含本 README 的目录。包内镜像为一个独立 Docker 归档；原始数据、源码和权重均属于本模型，数据准备不访问网络。

```bash
cd Qwen3-VL-8B
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 准备数据集和预训练权重

```text
weights/qwen3-vl-8b-instruct/  # 四个分片、索引、tokenizer、processor 和版本标记
sources/LLaMA-Factory/data/dataset_info.json
sources/LLaMA-Factory/data/mllm_demo.json 及对应图片
```

`sources.lock.json` 固定本模型源码和补丁。本模型使用锁定的LLaMA-Factory源码及基于官方NPU环境制作的固定交付镜像。 迁移到新目录后重新运行本模型的 `bash test/prepare_data.sh`。

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

本次功能配置：**完整官方 8B 权重；2 卡 BF16 LoRA，rank8，mllm_demo 六条图文样本，10 次优化更新**。输出 `adapter_model.safetensors + checkpoint-10/`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 完整数据训练：DriveLM-nuScenes v1.1

数据来自 [DriveLM 官方说明](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md)，文件身份固定在 [drivelm.sources.lock.json](drivelm.sources.lock.json)。以下是本项目为 Qwen3-VL 制定的六相机 LoRA 协议；没有将它写成 DriveLM 官方 llama-adapter 基线或官方排行榜结果。

### 放置完整原始文件

```text
data/drivelm/
├── raw/
│   ├── v1_1_train_nus.json
│   └── v1_1_val_nus_q_only.json
└── archives/
    ├── drivelm_nus_imgs_train.zip
    └── drivelm_nus_imgs_val.zip
```

官方百度下载入口如下；下载后按上面的目录放置，文件大小与摘要由脚本自动核验。

| 文件 | 官方下载入口 | 提取码 |
| --- | --- | --- |
| `v1_1_train_nus.json` | [训练问答](https://pan.baidu.com/s/1Vojg73jviguki0yvAB6nUg?pwd=73s8) | `73s8` |
| `drivelm_nus_imgs_train.zip` | [训练图像](https://pan.baidu.com/s/11xvxPzUY5xTIsJQrYFogqg?pwd=mk95) | `mk95` |
| `drivelm_nus_imgs_val.zip` | [官方验证图像](https://pan.baidu.com/s/1mUbHmtc4c5WhUQACW8_Blw?pwd=54vz) | `54vz` |
| `v1_1_val_nus_q_only.json` | [官方验证问题](https://pan.baidu.com/s/1p01Szh5QTtSAzSXdhLCTwQ?pwd=h9hi) | `h9hi` |

四个官方文件合计约 4.39 GB。训练图像包包含 24,432 张图像，验证图像包包含 4,794 张图像；Qwen 的 DriveLM 任务使用这两个官方子集，不必等待完整 nuScenes 传感器归档。原始文件保留，数据准备脚本解压、逐图解码及计算摘要，不访问网络。

**旧网盘 ZIP 是 2026-10-18 的功能资产包。** 本次新增数据和代码未打入旧包。以当前 GitHub 本模型目录作为工作目录，补入旧包的 `images/`、`weights/`、`sources/`，再放置上面四个数据文件；不要把旧包的脚本、配置或 `package_manifest.json` 覆盖到新源码目录。首次导入镜像可执行：

本轮实测机器的默认 Docker 分区空间不足，因此使用本项目独立的 Docker 存储。**在本轮机器复现时**，先在当前终端执行下面一行，再运行镜像、setup 和训练命令：

```bash
export DOCKER_HOST=unix:///run/migration-full8-20261019/docker.sock
```

该端点的存储为 `/var/aispace/lzd/migration_full_20261019/docker-data`，进程及启动配置由本轮现场维护；机器重启后需先确认端点可用。其他服务器使用其正常 Docker 服务即可，仍须核对固定 image ID 和存储容量。

```bash
sha256sum images/qwen.tar.gz
# 应为 0b0ce68cac12a215ad540ee9f458422cde3474a69810e5cd55fc2bb5f0c5faed
docker load --input images/qwen.tar.gz
bash test/setup.sh
```

setup 会核对本页固定 image ID 并准备源码；随后执行完整数据准备：

```bash
bash test/prepare_full_data.sh
```

这一步在固定镜像内检查原始文件、所有图像、场景划分及 **377,955 条有效问答的原生 SFT 编码**。会花费一段 CPU 处理时间，不启动 NPU 训练。输出在本模型 `data/drivelm/` 下；迁移目录后须重新准备，不能沿用旧绝对路径。

### 数据划分和完整性判据

| 项目 | 本项目完整数据协议 |
| --- | --- |
| 官方训练文件 | 696 场景、4,072 关键帧、377,956 条原始问答 |
| 空答案 | 官方文件中 1 条答案为空，明确排除并记录原始 ID，不补造标签 |
| 本地训练 / 留出 | 按固定 SHA256 排序划分 626 / 70 个场景；340,119 / 37,836 条有效问答；场景不重叠 |
| 输入 | 每条原始问题及六相机图像，固定相机名称和顺序；监督目标为原始答案 |
| 上下文 | 377,955 条原生编码均通过；最短 1,574、最长 2,629 token；入口使用 4,096，不沿用演示上限 2,048 |
| 官方验证文件 | 149 场景、799 帧、4,794 张图像、15,480 条原始条目，其中 1 条原始问题为空并保留标记 |
| 评测边界 | 37,836 条本地留出数据用于 loss；官方验证答案未公开，不产生或推算官方隐藏评分 |

转换生成单独的 `prepared/dataset_info.json`，不修改上游演示数据注册表。正式训练前重新核验官方归档及已解压图像、逐行对照官方问答、检查场景划分和上下文审计与当前文件摘要一致；改名六条演示数据或传入任意已有目录均不能通过。

### 双卡预检、完整训练及独立重载

```bash
# 真实驾驶数据预检：4 个训练场景、4 类问题各 4 条，双卡 8 次更新及重载评测
bash test/train_drivelm_preflight_2p.sh --device_id=0,2

# 正式三轮训练，使用新输出目录，从完整基座重新开始
bash test/train_full_2p.sh --device_id=0,2

# 待正式训练结束后，单卡重新加载适配器，评测全部 37,836 条本地留出问答
bash test/validate_full_1p.sh --device_id=0
```

正式配置见 [qwen3vl8b_drivelm_npu.yaml](qwen3vl8b_drivelm_npu.yaml)：两卡 BF16、LoRA rank8、语言线性层、每卡 batch=1、梯度累积 8、学习率 1e-4、cosine、warmup=0.1、三轮、梯度检查点，沿用上游默认冻结视觉骨干及投影层。数据准备阶段确定的上下文上限由入口写入实际命令。每轮保存完整优化器和两个 rank 的 RNG；保留最近两份 epoch 检查点。

本协议三轮预期 **63,774 次优化更新**。`full_training_check.json` 需要同时通过实际步数、epoch、最终模型/优化器有限值、LoRA 更新、两 rank 状态及完整本地留出 loss 检查；`full_validation_result.json` 还要求单卡独立重载后的完整留出 loss 与训练结束评测一致（rtol=atol=0.005）。`exp_of_trainer_mean_eval_loss` 仅是原生 Trainer 平均 loss 的指数，不当作 DriveLM 任务分数。

预检输出独立保留，正式训练重新创建 `test/output/full_<时间>/`，不从八步预检的短学习率计划续接。原 `validate_1p.sh` 用于十步演示配置；完整任务使用这里的 `validate_full_1p.sh`。

### 当前实测状态（2026-10-19）

官方完整训练与验证图像、问答及全部 377,955 条原生编码已检查通过；真实驾驶数据的双卡八步训练、模型/优化器检查及独立重载评测通过。完整预处理已完成，三轮正式训练的第 10 / 20 步日志 loss 分别为 3.2349693298 / 2.8300830841，保留在[启动阶段实机证据](validation/drivelm_complete_data_20261019.json)。2026-10-19 23:37（北京时间）[进度快照](validation/full_training_progress_20261019.json)已推进到 **2,111 / 63,774** 次更新；第 2,110 步训练日志 loss=0.161。训练正在进行，以上训练 loss 不作为完整留出或收敛验收。最终完整训练、完整留出重载和官方隐藏评分在未取得对应结果前均不标为通过。

# 训练结果展示

## 本项目实测（2026-10-12）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），不是官方训练系列。以下为本仓独立入口的实际结果。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | 完整官方 8B 基座；mllm_demo 6 条演示图文；另用独立合成图片检查内容 |
| 训练配置 | 2 张物理卡；BF16 LoRA SFT；10 次更新；20261012 固定镜像 |
| 保存状态 | 504 个适配器张量有限；252/252 个 LoRA B 非零；504 组优化器状态 step=10；2 rank 产物 |
| loss | 首条 3.2745351791，末条 2.5068483353；仅为十步记录，非收敛验收 |
| 独立验证 | 重新加载适配器，正确描述独立图片中的红色正方形；检查 1 张合成图 |
| 未完成 | 真实业务图文任务精度、完整训练、吞吐、续训及长稳 |

当前检查点 SHA256：`7ee34281193045e042789faa20a570fb6cf3ddc16df1c8aee0923c5640839c7e`。

执行时间（UTC）：`2026-10-12T10:43:46.947257+00:00` 至 `2026-10-12T10:47:42.245356+00:00`。原始执行命令：

```bash
bash test/train_2p.sh --device_id=0,2
bash test/validate_1p.sh --device_id=0
```

两条命令退出码均为 0。宿主卡号是当次资源分配记录；复现时按本页环境与卡号映射选择空闲设备。

证据：[执行与退出码](validation/entry_acceptance.json)、[训练摘要](validation/reference_run/train_result.json)、[模型/优化器检查](validation/reference_run/saved_state_check.json)、[独立验证数据](validation/reference_run/validation_result.json)。

另已完成[单镜像导入](validation/single_image_import.json)、[更换模型目录复验](validation/path_relocation.json)。[2026-10-12 原负向测试](validation/full_data_guard.json)仅证明当时指定演示数据被拒绝；[2026-10-19 回归](validation/full_data_guard_20261019.json)补充了改名和直接入口绕过检查。两者都不是正式数据正向训练验收。

## 通过判据

训练必须退出 0，并产生 `pass_functional_training`；独立验证必须退出 0，并产生 `pass_independent_functional_validation`。保存的模型/优化器、实际更新次数及上述推理范围均须检查通过。正式任务精度尚未测量的项目不填入估计值。

# 版本说明

## 变更

- 2026-10-19：补充完整 DriveLM 数据准备、场景留出、逐样本上下文检查及独立训练/评测入口；保留完整训练进行状态与最终验收的区别。
- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示 / 现象 | 处理 |
| --- | --- |
| 缺少镜像归档 / image ID 不一致 | 取得本模型完整包，核对 images.lock.json 与现有镜像用途 |
| 标注或图像路径缺失 | 重新运行本模型 prepare_data，核对 data 目录；不要生成假标签 |
| NPU 不空闲 / 卡号重复 | 核对宿主进程和物理映射，处理占用后再启动 |
| 只有训练日志没有通过报告 | 查看本次输出最后异常与退出码，不能据此宣称通过 |
| 选卡容器的 npu-smi 报 `ret is -9005` | 本轮环境报告保留该监控告警；宿主板卡查询正常、TorchNPU 识别两卡且真实训练通过。核对 environment.json 的 failures 与宿主状态，不将监控告警隐藏为完全无告警 |
| 迁入另一目录后文件路径错误 | 先 setup/prepare_data，再检查 plan，使用当前目录的新配置 |

本模型实机证据随验证报告提供；问题档案和内部操作记录留在本机，不随离线包上传。
