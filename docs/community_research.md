# 社区迁移调研（调研 2026-10-10，实测更新 2026-10-11）

只引用项目维护方的代码、文档或 CI。公开材料中的 A2/A3 与 8 卡结果不能直接外推为本项目 910C 2/4 卡通过。

| 模型 | 可复用的社区实现 | 公开证据及本项目缺口 |
| --- | --- | --- |
| BEVFormer | [Ascend DrivingSDK BEVFormer](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md) 固定 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) 的 `66b65f3` 并提供 NPU 补丁和训练脚本 | DrivingSDK 公布 A2/A5 8 卡结果，未看到本项目 910C 2 卡结果；需校验 2 卡全局 batch、学习率和 NDS/mAP。 |
| Sparse4D | [Ascend DrivingSDK Sparse4D](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md) 固定 [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 的 `c41df4b`，含 MMCV、mmdet、模型补丁 | 公开结果为 A2 8 卡；原配置的 `total_batch_size=48`、`num_gpus=8` 必须在 2 卡时一起调整。 |
| Qwen3-VL-8B | [Qwen 官方仓](https://github.com/QwenLM/Qwen3-VL)给出模型与微调代码；[LLaMA-Factory 示例](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/examples/README.md)同时给出 Qwen3-VL LoRA SFT 和 NPU 设备选择，[同提交的 A3 镜像说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)给出配套版本 | [verl 的昇腾 CI](https://github.com/verl-project/verl/blob/main/.github/workflows/nightly_ascend.yml)有 Qwen3-VL-8B 的 NPU GRPO/FSDP2 测试，但训练目标和 8 卡 A2 环境与本项目 2 卡 LoRA SFT 不同；仅作为可行性旁证。 |
| WorldEngine | [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine) 官方仓及 [AlgEngine 训练指南](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/algengine_usage.md) | 官方发布的是 CUDA 训练/SimEngine 流程。未找到项目维护方发布的完整 Ascend/910C 适配；本仓只迁移 AlgEngine VADv2 入口和已定位的变形注意力调用。 |
| Apollo VisionNet | [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net) 官方 BEV+OCC 训练仓和 [训练说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/run.md) | 官方依赖 PyTorch 1.10、CUDA、旧 MMCV/MMDetection3D；[部署仓](https://github.com/ApolloAuto/Apollo-Vision-Net-Deployment)为 TensorRT，不能替代 NPU 训练证明。未找到官方 910C 迁移。 |

DrivingSDK 当前维护的 `v26.1.0` 示例和安装指南见 [官方 README](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/README.md) 与 [安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)。本仓的 WorldEngine/Apollo 补丁参考其 `mx_driving.multi_scale_deformable_attn` 调用形式。已在 1# 完成有限形状的前向/梯度数值检查、两模型四卡合成检查和官方真实样本的四卡 10 步训练；完整指标仍需验证。

2026-10-11 已重新核对来源；Sparse4D 上游 HEAD 更新至 `249ffbb`（仅 README 两行），官方补丁在新提交核验通过。DrivingSDK 采用已发布并实测的 v26.1.0，当前开发分支另行记录；详情见[版本核验](latest_versions.md)。

## 本仓实测与工作流

本次使用已安装的 `ascend-model-migration`、`ascend-migration-analysis`、`ascend-mmlab-install-suite` 和 `ascend-docker` 技能流程，按上游版本定位 CUDA 依赖、复用 DrivingSDK 的 Ascend OpenMMLab 环境、审计补丁并在隔离容器内验证。上表项目维护方材料是来源依据，本仓实测见[验证记录](validation.md)。

BEVFormer、Sparse4D 已在 1# 两卡公开 nuScenes mini 上训练，Qwen 已在两卡完成演示数据 LoRA 十步及保存检查；Apollo 官方 train/val 与 mini 交集（242 train / 162 独立 val）与 WorldEngine 官方 NAVTEST 子集，均通过公开 quickstart 入口完成四物理卡真实短训练。WorldEngine 使用独立派生镜像、固定官方 nuPlan devkit 提交和官方感知基座。Apollo 已使用固定官方仓自带的 DLA 深度预训练权重完成全部初始化张量匹配检查；其编译 FocalLoss 改为 MMDetection 已有的 PyTorch 公式，有限案例 CPU/NPU 前后向对比通过。WorldEngine 的 NAVTEST 复用只检查功能；Apollo 另检查全部 162 个独立验证帧的推理输出，但未计算正式精度指标。实测不包含完整数据精度或官方训练硬件支持验收，完整镜像清单见[环境指南](environment.md)。
