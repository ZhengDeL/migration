# 社区迁移依据

本页列本仓已经采用并固定的实现。来源核验日期见[固定版本](latest_versions.md)，原文及章节对照见[原仓指导格式对应](readme_reference_alignment.md)。上游公开成绩不作为本仓实测数据。

| 模型 | 采用的实现 | 本仓适配范围 | 锁文件 |
| --- | --- | --- | --- |
| [BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md) | Ascend DrivingSDK 模型示例及固定补丁 | 华为已有 NPU 迁移；本仓另用双卡 mini 入口验证 | [固定来源](../PyTorch/contrib/cv/detection/BEVFormer/sources.lock.json) |
| [Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md) | Ascend DrivingSDK 模型示例及固定补丁 | 华为已有 NPU 迁移；本仓双卡配置与原八卡结果分别记录 | [固定来源](../PyTorch/contrib/cv/detection/Sparse4D/sources.lock.json) |
| [Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | Qwen 官方 8B 权重、LLaMA-Factory NPU/A3 方案 | 本仓双卡 LoRA SFT 与独立图文检查 | [固定来源](../PyTorch/contrib/nlp/Qwen3-VL-8B/sources.lock.json) |
| [WorldEngine](../PyTorch/contrib/cv/others/WorldEngine/README.md) | OpenDriveLab WorldEngine 官方仓与 AlgEngine 指导 | 本仓 AlgEngine VADv2 适配；不包含完整 SimEngine 或 RL 迁移 | [固定来源](../PyTorch/contrib/cv/others/WorldEngine/sources.lock.json) |
| [Apollo-Vision-Net](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | ApolloAuto 官方训练仓与自带 DLA 初始化 | 本仓 Ascend 训练兼容补丁；TensorRT 部署不能替代训练验证 | [固定来源](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/sources.lock.json) |
| [LSTM-Argoverse](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | Argoverse 官方 LSTM 车辆轨迹预测基线 | 本仓 NPU/HCCL、完整数据 curriculum 及数值/续训验证 | [固定来源](../PyTorch/contrib/cv/others/LSTM-Argoverse/sources.lock.json) |
| [STGCN-Traffic](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | STGCN 论文作者 TensorFlow 仓与 hazdzz 社区 PyTorch 实现 | 本仓迁移社区 PyTorch 的 PeMSD7(M) 70/15/15 协议 | [固定来源](../PyTorch/contrib/cv/others/STGCN-Traffic/sources.lock.json) |
| [ResWorld](../PyTorch/contrib/cv/others/ResWorld/README.md) | mengtan00/ResWorld 原作者模型、数据与训练协议 | 本仓真实 mini、三项数值修复与四卡控制；正式全量训练待完成 | [固定来源](../PyTorch/contrib/cv/others/ResWorld/sources.lock.json) |
| [MobileNetV2](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) | Ascend ModelZoo-PyTorch MobileNetV2 官方案例 | 本仓固定原网络；功能训练与官方 V1 权重全量 val 分别报告 | [固定来源](../PyTorch/contrib/cv/classification/MobileNetV2/sources.lock.json) |

具体迁移修改、原仓链接、数据准备及实际操作命令分别保留在各模型 README。训练样本量、更新次数、误差、任务指标、实际结果 JSON 与未完成项统一进入[九模型验证结果](validation.md)，避免调研进度页和验收结果重复维护。
