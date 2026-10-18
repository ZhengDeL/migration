# 迁移指导的原仓依据与格式对应

核对日期：2026-10-19。固定源码按 Git 对象校验后读取，不把开发分支网页变化混入已验证环境。

## 1. 原指导如何组织

- [DrivingSDK BEVFormer 原指导](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/BEVFormer/README.md)：概述、支持模型、准备训练环境（昇腾／模型／数据／预训练权重）、开始训练、结果、版本说明（变更／FAQ）。
- [DrivingSDK Sparse4D 原指导](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/Sparse4D/README.md)：简介、代码实现、准备训练环境、准备数据集及权重、快速开始（训练／结果）、版本说明。
- [ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)：概述、参考实现、准备训练环境及数据、开始训练（单卡／八卡及参数）、训练结果展示、版本说明和公网地址说明。

各原文标题略有区别，均先说明模型与来源，再准备环境／数据／权重，随后给出运行命令和对应结果。本仓九个模型的 README 以相同核心顺序组织：**概述 → 准备训练环境 → 开始训练 → 训练结果展示 → 版本说明**。每个模型仍有自己的命令、数据与实测记录。

## 2. 九个模型的直接依据

| 模型 | 原仓训练／数据文档 | 格式依据 | 本项目入口与范围 |
| --- | --- | --- | --- |
| BEVFormer | [固定版本原文](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/BEVFormer/README.md) | DrivingSDK BEVFormer | 两卡 mini；沿用固定官方补丁，原八卡结果不转记。 |
| Sparse4D | [固定版本原文](https://gitcode.com/Ascend/DrivingSDK/blob/0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837/model_examples/Sparse4D/README.md) | DrivingSDK Sparse4D | 两卡 Sparse4Dv3 mini；卡数、全局 batch、学习率和 AdamW 配置单列。 |
| Qwen3-VL-8B | [固定版本原文](https://github.com/hiyouga/LLaMA-Factory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/examples/README.md) | ModelZoo 核心顺序 | LLaMA-Factory 多模态 LoRA SFT；两卡 BF16 功能训练。 |
| WorldEngine | [固定版本原文](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/algengine_usage.md) | ModelZoo 核心顺序 | AlgEngine VADv2 四卡 imitation 功能路径；SimEngine／3DGS／RL 不在迁移范围。 |
| Apollo-Vision-Net | [固定版本原文](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/run.md) | ModelZoo 核心顺序 | 官方 DLA 初始化、真实 mini/OCC 的四卡功能训练与留出验证。 |
| LSTM-Argoverse | [固定版本原文](https://github.com/jagjeet-singh/argoverse-forecasting/blob/27696a19b3753a4ad8dcae3ad9c6107c5e4e237c/README.md) | ModelZoo 核心顺序 | 原基础 LSTM、坐标函数及1/10/30 curriculum；本项目两卡 HCCL 启动。 |
| STGCN-Traffic | [固定版本原文](https://github.com/hazdzz/STGCN/blob/5cfa82c78984fc58fafe4841c35618c6647c4219/README.md) | ModelZoo 核心顺序 | hazdzz PyTorch 社区实现；PeMSD7(M) 70/15/15，区别论文作者 TensorFlow 协议。 |
| ResWorld | [固定版本原文](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/README.md) | ModelZoo 核心顺序 | 作者80M网络、三损失与原65帧评测；四卡mini，原12epoch完整训练未验收。 |
| MobileNetV2 | [固定版本原文](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md) | ModelZoo MobileNetV2 | 原ModelZoo网络与1000类输出；两卡FP32功能训练及V1完整验证，600epoch未完成。 |

## 3. 如何阅读本仓命令和结果

- `test/train_2p.sh`、`train_4p.sh`、`validate_1p.sh` 等名称对应本项目实际存在且已验证的入口，不伪称原仓同名脚本。原仓八卡／CUDA 命令及其目录不直接复制成两卡／四卡 Ascend 的可执行命令。
- 本模型 `sources.lock.json`、`images.lock.json` 固定源码与镜像；README 列出环境和准备步骤。GitHub 提供源码／指导，大镜像、数据和权重由对应完整离线 ZIP 提供。
- “训练结果展示”引用该模型自己的 JSON，记录真实数据规模、卡数、精度、步数／epoch、检查点和未完成项。CPU/NPU 数值门槛是本项目工程门槛，不冒称华为业务验收标准。
- 后四模型的 FULL_VALIDATION.md 保留完整训练、数值、续训、吞吐和稳定性细节；ResWorld 的 NUMERICAL_REPAIR.md 保留实际根因与失败历史。这些附录不替代主 README 的顺序操作指导。
- 本次调整主指导结构与来源标注。运行代码、数据／镜像锁、既定门槛和原始实测 JSON 保持；格式调整本身不表示新增一轮 NPU 训练。

## 4. 来源快照

[来源清单](readme_reference_sources.json)记录11份固定原文的仓库、commit、文档路径、SHA256及实际标题。文档引用以固定commit为准。原仓许可证与网络文件仍保留在各模型自己的目录中。
