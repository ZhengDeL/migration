# 新增模型：910C 迁移前核验

核验日期：2026-10-12。此页记录候选实现及迁移条件；这四个新增模型尚未完成本项目的 NPU 训练验收。

## 先确定具体模型

| 用户给出的名称 | 已核验的官方 Ascend 案例 | 数据 / 任务 | 当前结论 |
| --- | --- | --- | --- |
| LSTM | `PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch` | TIMIT 语音识别，LSTM + CTC | 有官方历史训练实现；需要确认是否是所需的 LSTM 任务 |
| Stgcn | `PyTorch/contrib/cv/pose_estimation/ST-GCN` | 骨架动作识别；默认 Kinetics-skeleton，另有 NTU RGB+D 配置 | 有官方历史训练实现；骨架识别和交通预测的 STGCN 是不同项目 |
| Mobilenet | `PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch` | ImageNet 图像分类，MobileNetV2 | 建议以 V2 为候选；V1 / V3 需按业务需要另选 |
| Resword | 尚未确定 | 尚未确定 | 当前检索未找到可核验的模型项目；需正确名称或项目地址 |

早期候选曾提到 LSTM 情感分类。核对官方模型文档后，确认这里的华为案例实际是语音声学模型。文本分类、时间序列预测不能直接套用它的代码和验收指标。

## 官方来源和版本

已从 [Ascend 官方 GitCode 仓库](https://gitcode.com/Ascend/ModelZoo-PyTorch)拉取源码，并用 Git 远程 HEAD 和本地 checkout 核对：

```text
repository: https://gitcode.com/Ascend/ModelZoo-PyTorch.git
revision:   e779117e8cfeba296a7105e26adac0741ab04768
commit time: 2026-10-12T11:22:12+08:00
```

模型文档与关键源码的 SHA256 见 [来源核验记录](../verification/additional_models_research_20261012/sources.json)。GitHub 上的官方旧副本可辅助阅读；此次版本依据为拉取的 GitCode 提交。

官方指导分别是：

- [LSTM](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/nlp/LSTM_ID0468_for_PyTorch/README.md)
- [ST-GCN](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/contrib/cv/pose_estimation/ST-GCN/README.md)
- [MobileNetV2](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)
- [官方迁移指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/docs/zh/model_migration/README.md)
- [官方训练指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/docs/zh/model_training/README.md)

### 已有社区实现的适用范围

| 项目 | 官方文档列出的历史配套 | 当前代码中需要复核的内容 |
| --- | --- | --- |
| LSTM + CTC | PyTorch 1.5 / 1.8 | Apex AMP、NpuFusedAdam、Kaldi 特征数据、双向 LSTM 和 NPU CTCLoss 的前向 / 反向 |
| ST-GCN | PyTorch 1.5 / 1.8 | Apex AMP、HCCL 分布式训练、骨架图拓扑和批归一化；华为代码已把图卷积的 einsum 改为 matmul |
| MobileNetV2 | PyTorch 1.11 / 2.1 / 2.2 / 2.3 / 2.4 | Apex AMP、NpuFusedSGD、torchvision 配套、深度可分离卷积、数据加载和学习率策略 |

官方当前的随版本演进列表包含 BERT-SQuAD、ESPnet2、ResNet50-CIFAR、HRNet；本页的三个候选不在该列表中。LSTM 和 MobileNetV2 的当前 README 也明确说明未在最新昇腾配套中适配验证。因此，这些案例可作为迁移依据，不能直接据此声称已支持本项目的 CANN 9.1 / TorchNPU 2.7.1 / 910C 环境。

## 候选镜像与机器状态

已有工程的下面这套环境可作为新增模型的第一套候选环境。是否采用，取决于新增模型的实际兼容性验证。

```text
tag:          local/migration-drivingsdk:26.1.0-a3-headless-20261010
architecture: arm64
image ID:     sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
CANN:         9.1.0
Python:       3.10.21
PyTorch:      2.7.1+cpu
TorchNPU:     2.7.1.post6
```

2026-10-12 20:37（北京时间）重新只读检查了 1#：ARM64，8 张物理卡 / 16 个逻辑设备，健康状态均为 OK；未发现 NPU 运行进程和运行中的 Docker 容器；候选镜像 ID 与现有锁文件一致。此检查只证明机器与镜像可供后续实验使用，未运行新增模型。

1# 的 Atlas 800I A3 推理系列属性仍需在后续验证说明中保留。实验能运行不能替代官方训练硬件支持声明。

## 每个模型的迁移工作

### LSTM

先明确任务：语音识别、文本分类、时间序列预测所需的数据处理、网络输出和评测方法不同。

如采用官方 LSTM + CTC：

1. 保留官方网络、特征维度、类别字典与 CTC 目标含义，固定来源与配置。
2. 准备 TIMIT 原始数据和 Kaldi 特征，检查 `fbank.scp`、`phn_text`、`units`；不把随机序列作为语音模型验收数据。
3. 验证 LSTM / CTCLoss 的 NPU 前向和反向，再验证完整模型的梯度与参数更新。
4. 先单卡，再双卡 HCCL；保存优化器、恢复 checkpoint，独立验证集评测音素识别结果。
5. 评测报告注明数据划分、解码方式和字典，不能照搬旧 README 的数值作为本项目结果。

### ST-GCN

如选择骨架动作识别，优先沿用华为已适配的原始 ST-GCN 网络和图卷积修改。

1. 固定 Kinetics 或 NTU 的一种协议。Kinetics 的默认配置是 OpenPose 18 个关节、400 类；NTU 使用另一套骨架拓扑及数据划分，不能只改分类数。
2. 核对 `train_data.npy` / `train_label.pkl` 与 `val_data.npy` / `val_label.pkl`。保证样本、标签数量和骨架 shape 一致，训练与验证样本不重叠。
3. 迁移 Apex AMP 和旧训练入口；先核验 FP32，再验证所需混合精度。
4. 双卡 HCCL 验证每个 rank 的前向、梯度、参数更新和同步；保存并重新加载 checkpoint。
5. 独立单卡评测 Top-1 / Top-5，明确所用数据协议和子集范围。没有真实骨架数据时仅能做算子功能测试。

如选择交通预测，需要改选交通 STGCN 上游和数据，重新评估；不复用骨架识别案例的训练或精度结论。

### MobileNetV2

1. 固定 V2 网络、宽度倍率、分类头、图像预处理和上游配置。
2. 数据按 ImageNet 的 `train/<class>` / `val/<class>` 布置，训练与验证共享类别映射；先用可追溯的真实图片子集做功能验证。
3. 核验 depthwise / grouped convolution、BatchNorm、ReLU6 的 NPU 前向和反向，迁移旧 Apex 训练依赖。
4. 单卡 / 双卡完成优化器更新、checkpoint 保存 / 加载和独立验证。
5. 完整 ImageNet 精度、吞吐和长时间稳定性单独验收。合成图像和小规模图片子集不能证明 ImageNet 精度。

### Resword

名称待确认，尚未选择上游、数据或网络结构。即使最终是 ResNet，也需要明确 ResNet18 / 50 等版本和任务后再落实代码。

## 独立工程交付要求

名称、任务和版本确定后，继续沿用现有交付规则，每个模型单独具备：

- `README.md`：明确镜像、数据、逐步操作和成功判据。
- `sources.lock.json` / `images.lock.json`：固定真实提交、镜像 digest / ID、关键资产校验值。
- `test/`：自己的环境检查、数据准备、训练、完整训练和独立验证入口。
- `validation/`：本模型的真实执行证据，分别记录通过、失败及未验证范围。
- 完整可用目录及其独立归档：包含自己的镜像、必要数据、权重和校验清单，不依赖其他模型目录。

当前状态是来源调研完成、模型选型待确认。未新增未经验证的训练脚本，未把这四项写为迁移完成，也未重新封装已验收的五个模型交付包。
