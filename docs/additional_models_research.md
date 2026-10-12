# 新增智驾模型：来源、选型与验证进展

核验日期：2026-10-12。四项按智驾用途组织。本页记录当前阶段；各项完整数据训练、正式精度、性能和长期稳定性均尚未验收。

## 当前任务与来源

| 名称 | 本轮采用或研究的实现 | 用途 | 当前验证范围 |
| --- | --- | --- | --- |
| LSTM | Argoverse 官方基础 LSTM，建议工程名 LSTM-Argoverse | 用车辆历史二维轨迹预测未来位置 | 原始编码器 / 解码器单 NPU 前向、反向、一次参数更新及 CPU 对照通过；真实数据及完整工程未验收 |
| STGCN | hazdzz/STGCN 的 PyTorch 交通实现，配合原论文作者的 PeMSD7 数据 | 道路传感器时间序列预测；本轮预测交通速度 | PeMSD7-M 功能子集双卡 10 步，checkpoint 独立重载及 241 个留出窗口验证通过 |
| ResWorld | 用户指定的 mengtan00/ResWorld | 多相机时序输入的端到端智驾规划 | 完整官方网络构造通过，80,766,238 参数；前向、反向、训练及真实数据验收待完成 |
| MobileNetV2 | Ascend ModelZoo 的 V2 网络，保留 1000 类输出头 | 图像分类基础网络；可作为智驾视觉骨干参考 | 公开 Imagenette 图片子集双卡 10 步，checkpoint 独立重载及 40 张留出图片验证通过 |

LSTM 的进一步研究见 [独立研究说明](lstm_driving_research.md)及 [实际单 NPU 复验](../verification/lstm_driving_research_20261012/README.md)。其他三项的当前阶段摘要见 [progress.json](../verification/additional_models_research_20261012/progress.json)。MobileNetV2 / STGCN 的本地独立工程仍在完善指导和封包，尚未作为最终离线交付发布。

## 固定源码

| 实现 | 官方或作者来源 | 本轮提交 |
| --- | --- | --- |
| LSTM 官方预测基线 | [jagjeet-singh/argoverse-forecasting](https://github.com/jagjeet-singh/argoverse-forecasting) | 27696a19b3753a4ad8dcae3ad9c6107c5e4e237c |
| STGCN PyTorch 实现 | [hazdzz/STGCN](https://github.com/hazdzz/STGCN) | 5cfa82c78984fc58fafe4841c35618c6647c4219 |
| STGCN 原论文作者实现 / 数据 | [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18) | 4ab75ab69af354d6a0ab86aefa082d389b187127 |
| ResWorld 作者实现 | [mengtan00/ResWorld](https://github.com/mengtan00/ResWorld) | a5ae973001dd6001d95b3389020a9e263d73a56e |
| MobileNetV2 华为案例 | [Ascend/ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) | e779117e8cfeba296a7105e26adac0741ab04768 |

STGCN 的原作者实现是 TensorFlow；本轮 PyTorch 网络采用注明来源的社区实现。ResWorld 的拼写及项目已按用户给出的链接固定。LSTM 上游提交日期为 2020-05-29，本轮 API 核验时仍是 master；它是一套明确的基础案例。

## 华为已有案例的适用范围

最初检索保存的 [sources.json](../verification/additional_models_research_20261012/sources.json)保留 ModelZoo 原文及关键源码的 SHA256，用于来源对比。

- ModelZoo 的 LSTM_ID0468_for_PyTorch 使用 TIMIT 语音和 CTC。它提供历史昇腾实现，但任务与当前车辆轨迹预测不同。
- ModelZoo 的 ST-GCN 使用骨架动作识别。交通 STGCN 的道路图、输入、数据和指标需要独立迁移与验证。
- MobileNetV2 官方案例可提供网络和训练流程依据；其 README 明确说明未在最新配套中适配验证，本项目需以实际测试为依据。
- 官方 [迁移指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/docs/zh/model_migration/README.md)和 [训练指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/docs/zh/model_training/README.md)作为工程结构和训练适配依据。

这些历史案例的原有精度数值不作为本项目的测量结果。

## 本轮实测镜像

```text
tag:          local/migration-drivingsdk:26.1.0-a3-headless-20261010
architecture: arm64
image ID:     sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
CANN:         9.1.0
Python:       3.10.21
PyTorch:      2.7.1+cpu
TorchNPU:     2.7.1.post6
```

1# 为 Atlas 800I A3 推理系列。这里的实验结果不改变官方训练硬件支持范围。完整模型需分别固定自己的镜像和来源，提供独立 README、数据准备、训练、评测与归档；各模型不会共享一个不可拆分的训练入口。

## 后续验收

- LSTM：验证真实 Argoverse 1 数据、归一化和坐标恢复、双卡 HCCL、checkpoint 与 ADE / FDE / Miss Rate。
- STGCN：完整 PeMSD7-M 或另行固定 METR-LA / PEMS 协议后验收；功能子集不能证明论文精度。
- ResWorld：准备完整数据与官方配置需要的监督、验证 NPU 网络前向 / 反向、训练重载与规划评测。
- MobileNetV2：完整 ImageNet 分类训练及指标；Imagenette 子集没有证明下游检测、分割或智驾业务精度。

内部问题台账、失败排查日志和操作记录留在本机。最终交付只包含每个模型必要的源码、指导、镜像、数据、权重、验证结果及校验清单。
