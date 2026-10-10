# 社区迁移调研（2026-10-10）

只引用项目维护方的代码、文档或 CI。公开材料中的 A2/A3 与 8 卡结果不能直接外推为本项目 910C 2/4 卡通过。

| 模型 | 可复用的社区实现 | 公开证据及本项目缺口 |
| --- | --- | --- |
| BEVFormer | [Ascend DrivingSDK BEVFormer](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md) 固定 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) 的 `66b65f3` 并提供 NPU 补丁和训练脚本 | DrivingSDK 公布 A2/A5 8 卡结果，未看到本项目 910C 2 卡结果；需校验 2 卡全局 batch、学习率和 NDS/mAP。 |
| Sparse4D | [Ascend DrivingSDK Sparse4D](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md) 固定 [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 的 `c41df4b`，含 MMCV、mmdet、模型补丁 | 公开结果为 A2 8 卡；原配置的 `total_batch_size=48`、`num_gpus=8` 必须在 2 卡时一起调整。 |
| Qwen3-VL-8B | [Qwen 官方仓](https://github.com/QwenLM/Qwen3-VL)给出模型与微调代码；[LLaMA-Factory 示例](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/examples/README.md)同时给出 Qwen3-VL LoRA SFT 和 NPU 设备选择，[同提交的 A3 镜像说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)给出配套版本 | [verl 的昇腾 CI](https://github.com/verl-project/verl/blob/main/.github/workflows/nightly_ascend.yml)有 Qwen3-VL-8B 的 NPU GRPO/FSDP2 测试，但训练目标和 8 卡 A2 环境与本项目 2 卡 LoRA SFT 不同；仅作为可行性旁证。 |
| WorldEngine | [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine) 官方仓及 [AlgEngine 训练指南](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/algengine_usage.md) | 官方发布的是 CUDA 训练/SimEngine 流程。未找到项目维护方发布的完整 Ascend/910C 适配；本仓只迁移 AlgEngine VADv2 入口和已定位的变形注意力调用。 |
| Apollo VisionNet | [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net) 官方 BEV+OCC 训练仓和 [训练说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/run.md) | 官方依赖 PyTorch 1.10、CUDA、旧 MMCV/MMDetection3D；[部署仓](https://github.com/ApolloAuto/Apollo-Vision-Net-Deployment)为 TensorRT，不能替代 NPU 训练证明。未找到官方 910C 迁移。 |

DrivingSDK 当前维护的 `v26.1.0` 示例和安装指南见 [官方 README](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/README.md) 与 [安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)。本仓的 WorldEngine/Apollo 补丁参考其 `mx_driving.multi_scale_deformable_attn` 调用形式；这属于从相似 BEV 算子推导的适配，尚需 NPU 数值与梯度回归。
