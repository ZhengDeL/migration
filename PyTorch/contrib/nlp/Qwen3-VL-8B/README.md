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

沿用固定 LLaMA-Factory 的 NPU 多模态 SFT 与 LoRA 实现，以本目录两卡 BF16 入口管理演示数据、保存和独立图文重载。

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

## 完整数据训练

2026-10-19 已确定正式任务为 **DriveLM-nuScenes v1.1 全量驾驶问答**，来源为 [DriveLM 官方准备说明](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md)。官方完整 nuScenes 可复用为六相机图像源。训练问答 `v1_1_train_nus.json` 为 192,961,829 bytes，固定 SHA256 为 `ecd3c13e9beb27295bcf6182ea35fff7c1d6a9ba58d40e48fb38d713094759fd`。

**正式入口暂未开放。** 旧实现只拒绝名为 `mllm_demo` 的数据集，改名或传入已有目录不能证明数据完整。当前宿主、模型及直接训练入口均在更改结果指针和启动容器前拒绝 full 模式；五项负向回归在本地和服务器 ARM64 Linux 通过，见 [门禁修复实测](validation/full_data_guard_20261019.json)。这项修复不代表正式训练已完成。

后续需完成真实文件身份与媒体覆盖检查、场景划分、ShareGPT 转换、六相机输入的全部样本上下文长度检查，以及双卡训练和独立评测。原演示配置的 2,048 token 上限不能未经核验用于正式六相机任务。官方 val 只公开问题，隐藏答案的排行榜评分需要官方评测；本地留出集指标须单独标识。当前 `validate_1p.sh` 仅覆盖已实测的功能配置。

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

- 2026-10-19：确认 DriveLM 正式任务，修复改名演示数据的 full 入口校验缺口，明确正式训练未开放的范围。
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
