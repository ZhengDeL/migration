# LSTM 智驾选型与 Ascend 910C 迁移研究

研究日期：2026-10-12。已完成官方候选比较、源码固定和单个 NPU 上的原始神经网络功能检查。真实数据训练、双卡通信及轨迹精度尚未验收；完整 LSTM 智驾训练工程尚未交付。

## 建议选型

建议以 **Argoverse 官方 Motion Forecasting Baselines 中的 LSTM 车辆轨迹预测**作为第一套迁移案例，工程名称使用 `LSTM-Argoverse`，把版本、任务与数据固定下来。

官方来源：[jagjeet-singh/argoverse-forecasting](https://github.com/jagjeet-singh/argoverse-forecasting)。本轮已拉取源码，并通过 GitHub 官方 API 核验当前 `master` 为：

```text
27696a19b3753a4ad8dcae3ad9c6107c5e4e237c
```

该提交时间为 **2020-05-29**，在本轮核验时仍是上游 `master`。它是成熟的官方基础案例。选择它是为了保持明确、可复现的 LSTM 网络与数据协议；本文不把它描述为当前最先进的轨迹预测模型。

该仓库明确提供 LSTM、LSTM + social、LSTM + map 三种轨迹预测基线。首轮选择基础 LSTM，使用车辆历史二维位置；之后再单独评估 social / map 变体。基础版和地图版的输入、预处理和评测协议不同，指导中会分别写明。

## 具体预测什么

| 项目 | 官方基线的设置 |
| --- | --- |
| 任务 | 预测目标车辆未来的二维轨迹 |
| 数据 | Argoverse 1 Motion Forecasting |
| 历史窗口 | 20 个观测位置，10 Hz，约 2 秒 |
| 预测窗口 | 30 个未来位置，10 Hz，约 3 秒 |
| 基础输入 | `X, Y`；需要按上游规则处理位移、平移和旋转归一化 |
| 网络 | LSTMCell 编码器与自回归 LSTMCell 解码器 |
| 神经网络默认大小 | 位置 embedding 8 维，hidden state 16 维，输出 2 维 |
| 评测 | 使用独立验证集，基础版固定单条预测 K=1，报告 ADE、FDE、Miss Rate；地图变体另有地图相关指标 |

原始代码中 `EncoderRNN` / `DecoderRNN` 位于 `lstm_train_test.py`，基础特征定义在 `utils/baseline_config.py`。数据采样频率见 [Argoverse 1 官方说明](https://www.argoverse.org/av1.html)。数据输入是目标车辆的轨迹坐标。

ADE 是整个预测窗口的平均位置误差，FDE 是末帧位置误差，恢复城市坐标后按米计算。Miss Rate 应使用固定的 2 米末帧误差阈值及 K 值；基础版 K=1 与多候选地图版的 minADE / minFDE 需要分开报告。

## 为什么优先选它

1. 来源是数据集的官方预测基线仓库，训练、推理、评测入口和参数有明确说明。
2. 输入输出直接对应智驾车辆轨迹预测，能形成清楚的数据及验收协议。
3. 基础网络使用 PyTorch 原生 Linear、ReLU、LSTMCell 等组件；神经网络代码没有额外 CUDA 扩展。本轮已验证这两个网络类的 NPU 数学功能。
4. 官方提供完整数据和功能测试用的 sample 数据指引，适合先验证小规模流程，再做完整数据验收。

仓库列出的历史依赖包括 PyTorch 1.2.0、Argoverse 1.0.0、NumPy 1.17.1 等。适配 910C 时需要重新验证现代依赖配套；不能直接把旧版 `requirements.txt` 安装进当前 CANN / TorchNPU 镜像并认为兼容。

## 其他可靠候选

| 候选 | 来源 | 适合的用途 | 此阶段判断 |
| --- | --- | --- | --- |
| Argoverse LSTM baseline | [官方基线](https://github.com/jagjeet-singh/argoverse-forecasting) | 基础车辆轨迹预测；可扩展 social / map 特征 | 优先作为首个可复现迁移案例 |
| Convolutional Social Pooling / CS-LSTM | [论文作者代码](https://github.com/nachiket92/conv-social-pooling) | 考虑邻车交互及驾驶行为的车辆轨迹预测 | 有原始论文和作者实现；代码基于 PyTorch 0.4，迁移工作与数据准备需单独评估 |
| STA-LSTM / Attention-LSTM | [论文作者 VTP 仓库](https://github.com/leilin-research/VTP) | 时空注意力、混合交通环境中的车辆轨迹预测 | 有作者实现和 NGSIM 数据指引；属于不同网络变体，不能共用基础 LSTM 的验收结论 |

对这些候选，本轮尚未核验到华为社区提供的对应 Argoverse / CS-LSTM 完整 910C 训练案例。此前查到的 ModelZoo `LSTM_ID0468_for_PyTorch` 实际是 TIMIT 语音 + CTC 案例，不能作为这些智驾网络已经迁移成功的依据。

社区核查范围为 Ascend ModelZoo 固定源码、本项目已有 DrivingSDK 资料及官方仓库检索，日期为 2026-10-12。DrivingSDK 的 [DenseTNT 指导](https://github.com/Ascend/DrivingSDK/blob/master/model_examples/DenseTNT/README.md)提供轨迹预测的昇腾训练流程参考；该页已标注停止随版本演进，其网络、Argoverse 2 数据和历史 A2 测试结果与本次 LSTM 不同。未发现匹配案例是一项有范围的检索结果。

## 迁移时必须保持的内容

- 固定 Argoverse 1 协议、官方源码提交及基础网络变体。
- 保持 20 个历史位置与 30 个未来位置的含义，核对采样间隔、坐标系和物理单位。
- 保持上游的归一化、位移表示、自回归解码和逆变换逻辑，检查恢复后输出的真实坐标。
- 上游训练采用 Adam / MSE，自回归输入使用自己的预测；包含 1、10、30 步的训练 rollout 阶段。完整迁移要核对这些训练规则，不能只把本轮单次数学检查扩大为训练复现。
- 训练 / 验证按官方 sequence 划分；验证未来轨迹不能进入训练。测试集缺少公开真实未来轨迹时，不以它计算本地精度。
- 明确替换旧 CUDA 设备逻辑，验证 LSTMCell 前向、反向和优化器更新，再验证完整数据处理及评测流程。
- checkpoint 保存网络、优化器、配置和进度；在独立路径重载，核对结果与数据摘要。
- 单卡验证后再做双卡 HCCL，保留各 rank 的梯度、参数更新及同步证据。

## 2026-10-13 真实数据工程实测

已提供 [LSTM-Argoverse 独立工程和使用指导](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md)。从校验过的官方完整 train/val 归档选取 128/32 条真实轨迹，双物理卡各 10 步训练、HCCL 参数一致性、checkpoint 独立严格重载和 CPU 数值对照通过，最大绝对误差 `5.960464477539063e-08`。32 条留出轨迹实际 ADE=13.7612 m、FDE=26.4804 m、Miss Rate=1.0；随机初始化的功能运行质量未达到任务精度验收。

原始 CSV 时间间隔并非每条都恒定 0.1 秒；保留实际时间戳，核对有限且严格递增并记录间隔，没有重采样。固定配套的原生 LSTMCell 在 NPU batch=1 有数值不稳定，本模型采用独立样本批维补齐并在 loss / metrics 前裁剪；真实输入下原始 CPU 对照及 14 个参数梯度控制通过。具体证据和范围见独立工程 `validation/lstm_batch1_control.json`。完整 1/10/30 步 curriculum 未验收。

## 2026-10-12 早期可行性检查

本轮提取了官方源文件中原始 `EncoderRNN` 和 `DecoderRNN` 类，使用合成二维坐标测试 NPU 前向、反向、一次优化器更新及 CPU 输出对照。2026-10-12 21:54（北京时间）还通过了随本次提交提供的复验脚本。

| 检查 | 1# 实测结果 |
| --- | --- |
| 数据与精度 | 合成坐标，batch=2，FP32，seed=42 |
| 输入 / 输出 | `[2,20,2]` / `[2,30,2]` |
| CPU / NPU 同权重前向对照 | 最大绝对误差 `4.470348358154297e-08`；阈值 rtol / atol 均为 `0.001` |
| 梯度 | 14 个参数张量均取得有限 NPU 梯度 |
| 优化器更新 | 1 次 Adam 更新，14 个参数张量发生变化 |
| 完整数据流程 / 双卡 / ADE、FDE、Miss Rate | 未验证 |

实测使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010`，CANN 9.1.0、PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post6；完整 image ID、固定源码、实际结果 JSON 与复验命令见 [研究证据](../verification/lstm_driving_research_20261012/README.md)。此处 `+cpu` 是基础 PyTorch wheel 的版本后缀，实际网络、loss 和梯度均由复验脚本检查为 NPU 张量。

本轮不包含真实 Argoverse 样本、官方特征处理及业务指标。该结果不能替代整个模型迁移、完整训练或智驾精度验收。

当前独立工程已经提供自己的 README、镜像与来源锁、数据准备、训练和独立验证。上表是最初神经网络探针的历史范围，2026-10-13 的真实数据进展见前节。1# 的推理系列硬件属性仍需保留，实验结果不能替代官方训练硬件支持范围。
