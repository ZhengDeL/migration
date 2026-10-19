# NPU 实测性能与 GPU 对照状态

核对日期：**2026-10-19**。本轮范围为八个模型；WorldEngine AlgEngine 由独立仓验收。

## 当前结论

四个模型已有 NPU 训练吞吐记录；**这八个模型均没有同条件的原生 CUDA/GPU 实测基线，目前无法给出比 GPU 快或慢多少的比例**。CPU/NPU 数值一致性、任务精度和训练完成情况见[模型验证结果](validation.md)，分别验收。

用户已确认当前没有可用 GPU 机器。本页先补充[作者与官方公开参考](#已核验的公开-gpu-参考)，同条件实测待机器确定后执行。

下表摘录已经公开的原始 JSON，保留原测试日期与配置。LSTM、STGCN、MobileNetV2 的计时区间只有约 0.6～2.6 秒，且每种配置仅执行一次，因此这些数值只能作为初步吞吐记录，不能作为稳定性能或硬件选型结论。

## 已记录的 NPU 训练吞吐

单位为 **samples/s，全局训练样本/秒**。每个样本分别指一条轨迹、一个交通时间窗口、一张图像、一个 ResWorld 训练样本。它们不是推理 FPS，也不能跨模型比较。

| 模型 | 实测数据范围 | 每 rank batch | 1 rank 吞吐 | 多 rank 吞吐 | 原生 GPU 吞吐 / 差异 | 原始证据 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| LSTM-Argoverse | 官方 Argoverse 功能子集，128 train / 32 val；不是完整训练集 | 8 | 177.918 | 2 rank：364.430 | 未测 / 无法计算 | [完整 JSON](../PyTorch/contrib/cv/others/LSTM-Argoverse/validation/extended_acceptance_20261018.json) |
| STGCN-Traffic | PeMSD7(M) 全量时间划分，8,857 train 窗口；计时只覆盖下述更新数 | 16 | 1,282.998 | 2 rank：1,692.274 | 未测 / 无法计算 | [完整 JSON](../PyTorch/contrib/cv/others/STGCN-Traffic/validation/extended_acceptance_20261018.json) |
| MobileNetV2 | 公开 Imagenette 功能子集；不是完整 ImageNet train | 4 | 87.506 | 2 rank：153.102 | 未测 / 无法计算 | [完整 JSON](../PyTorch/contrib/cv/classification/MobileNetV2/validation/extended_acceptance_20261018.json) |
| ResWorld | 真实 nuScenes mini，原功能训练入口 | 1 | 0.375813 | 4 rank：1.206015 | 未测 / 无法计算 | [完整 JSON](../PyTorch/contrib/cv/others/ResWorld/validation/extended_acceptance_20261018.json) |

原始 JSON 的字段为 `benchmark_1p`、`benchmark_2p` 或 `benchmark_4p`；包含检查点 SHA256、数据范围、batch、计时、吞吐、延迟和本 rank 峰值已分配显存。表中数值仅作显示舍入，原始精度保留在 JSON。

### 计时协议与实际持续时间

四项均于 **2026-10-18** 执行，预热 10 更新，计时 50 更新。计时包含真实数据加载、主机至 NPU 传送、前向、反向、优化器及逐次同步，多 rank 使用同步后的耗时。每 rank batch 固定，因此从 1 rank 到多 rank 时全局 batch 同时增大；这些记录属于弱扩展测量。

| 模型 | 1 rank 计时总秒数 | 多 rank 计时总秒数 | 多 rank 单次更新 P50 / P95（秒） |
| --- | ---: | ---: | ---: |
| LSTM-Argoverse | 2.248227 | 2.195212 | 0.043665 / 0.045354 |
| STGCN-Traffic | 0.623540 | 0.945473 | 0.018857 / 0.021097 |
| MobileNetV2 | 2.285565 | 2.612637 | 0.051396 / 0.056475 |
| ResWorld | 133.044830 | 165.835480 | 3.335837 / 3.535822 |

例如 LSTM 双 rank 的吞吐为 `50 × 8 × 2 / 2.19521164894104`。表中 P50/P95 为训练更新延迟，不能解释为单个样本的响应延迟。

### 硬件、软件与测量限制

- ARM64 Atlas 800I A3，板卡 `IT22HMDA_4_S`，driver `26.1.1`、firmware `9.0.0.9.220`。每 rank 使用一张不同物理卡中的一个逻辑 NPU 芯片；**并未使用每张双芯片物理卡的全部芯片**。GPU 对照须同时记录物理卡数和计算芯片数。
- 固定镜像 `local/migration-drivingsdk:26.1.0-a3-headless-20261010`，image ID `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`；Python 3.10.21、CANN 9.1.0、PyTorch 2.7.1、torch_npu 2.7.1.post6、DrivingSDK 26.1.0.post2。
- FP32、确定性算法；各模型的具体精度开关和算子配置以其原始 JSON 为准。ResWorld 使用数值修复后的 GELU、时序差分、相机几何计算及训练注意力实现，并记录同步梯度规约方式；这些适配可能影响耗时，当前没有逐项性能归因实验。
- 其他模型任务共用主机 CPU、存储并使用不同物理 NPU；不是独占主机实验。单次短计时没有反映重复运行方差，也没有整机能耗对照。
- 2026-10-19 ResWorld 原 runner / GeoBEV / EMA 的两轮精确恢复验证使用每 rank batch=2。上面的旧吞吐使用 batch=1、功能入口，**不能转写成该原 runner 的性能结果**。
- LSTM 完整训练、MobileNetV2 完整 ImageNet 验证等后续结果，没有改变上述吞吐实验的数据身份。

## 其余四项与正在运行的训练

| 模型 | 当前性能证据 | 原生 GPU 对照 |
| --- | --- | --- |
| BEVFormer | 已有 mini 训练与重载功能记录，独立吞吐基准未完成 | 未测 |
| Sparse4D | 已有 mini 训练与重载功能记录，独立吞吐基准未完成 | 未测 |
| Qwen3-VL-8B | DriveLM 全量三轮训练正在执行；[带时刻的训练进度](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/full_training_progress_20261019.json)包含原生日志行 | 未测 |
| Apollo VisionNet | 已有 mini 训练与重载功能记录，独立吞吐基准未完成 | 未测 |

Qwen 日志中约 10.8 秒/optimizer update 是当前训练过程的观察值，包含其 batch、梯度累积、真实样本长度及运行环境影响；尚未形成独立性能基准。不能将它换写成已验收的 tokens/s、推理吞吐或 GPU 加速比。

## 已核验的公开 GPU 参考

以下为 **2026-10-19 检查的作者/官方发布数据，未由本项目复跑**。README 固定到具体提交，论文固定到版本或出版页；[来源与核对清单](performance_sources.json)记录定位、抓取日期、摘要和未披露字段。公开来源未提供的 batch、精度或计时边界标为未知，不自行补全。来源可靠性与实验条件是否相同是两项判断。

### 训练用时参考

| 模型及来源 | 作者硬件与条件 | 公开结果 | 与当前 NPU 记录的差异 |
| --- | --- | --- | --- |
| [LSTM-Argoverse 原仓 README](https://github.com/jagjeet-singh/argoverse-forecasting/blob/27696a19b3753a4ad8dcae3ad9c6107c5e4e237c/README.md#usage) | EC2 p2.8xlarge 主机，配有 8×K80、32 vCPU、488 GiB RAM；无 map/social 特征的 LSTM | train 约 5.5 小时；test 约 0.1 小时 | 主机配卡数不等于训练实际使用卡数，README 未逐项明确；不是本仓功能子集短计时。原仓早停/训练轮次、特征缓存和本仓双 rank curriculum 需核对后才能比较总耗时 |
| [STGCN 原论文 §4.3 / Table 3](https://www.ijcai.org/proceedings/2018/0505.pdf#page=6) | GTX 1080、Xeon E5-2620 v4；PeMSD7(M)，STGCN(Cheb)，RMSprop，50 epoch、batch=50 | 训练总耗时 272.34 秒 | 作者发布的 TensorFlow 实现与本仓选用的 hazdzz PyTorch 实现不同；数据划分、batch、训练停止条件和框架未对齐，不能与本仓 52 epoch 总时长或短吞吐直接求比 |
| [Sparse4Dv3 作者 README](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/README.md#results-on-validation-split) | 8×RTX 3090 24 GB；ResNet-50、ImageNet 初始化、256×704、100 epoch、nuScenes；关联配置全局 batch=48、每卡 6 | 训练约 22 小时 | 本仓当前记录为双卡 mini 十步，尚未完成相同完整数据/轮次。README 未在结果行完整披露精度、计时边界，不能按卡数线性折算 |

### 推理速度参考

| 模型及来源 | 作者硬件与配置 | 公开结果 | 使用边界 |
| --- | --- | --- | --- |
| [BEVFormer 论文 v2 Table 6](https://arxiv.org/html/2203.17270v2#S4.T6) | V100；R101-DCN、900×1600 输入、200×200 BEV、6 层、多尺度；nuScenes val | 1.7 FPS | 本仓训练使用 FP16 配置，未完成对应推理吞吐测量；论文该表未完整说明 batch/精度/计时边界。表中的 BEVFormer 模块 130 ms 不是完整网络延迟，不能据此计算整网 FPS |
| [Sparse4Dv3 作者 README](https://github.com/HorizonRobotics/Sparse4D/blob/249ffbb695f4e9db628d953e2bf6d36de04bbb69/README.md#results-on-validation-split) | ResNet-50、256×704；同一结果行列出推理 FPS | 19.8 FPS | README 的“8×3090”描述整体实验环境，不能据此推定 FPS 使用 8 卡；该行未完整说明推理卡数、batch、精度和计时边界，本次不补推定值 |
| [Apollo 官方 TensorRT 部署仓](https://github.com/ApolloAuto/Apollo-Vision-Net-Deployment/blob/a692c23c0d35bb5010163d4bd96a57badc1d4c73/README.md#benchmarks) | NVIDIA Orin，TensorRT，nuScenes，batch=1 | 5.7 FPS | 属于车端 TensorRT 推理，不是原生 PyTorch CUDA 训练；表格未注明 Orin 具体 SKU、功耗模式、精度及完整计时范围，不能与本仓 NPU 训练吞吐比较 |

### 未找到可采用数据的范围

| 模型 | 本次实际检查 | 结果与后续 |
| --- | --- | --- |
| ResWorld | [固定原仓 README](https://github.com/mengtan00/ResWorld/blob/a5ae973001dd6001d95b3389020a9e263d73a56e/README.md)，含质量表和四卡训练命令 | 未见训练吞吐/总耗时/推理 FPS；质量表中的 L2、碰撞率不属于硬件速度。保留待测 |
| Qwen3-VL-8B | [Qwen 官方微调说明](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/README.md) | 未见与本仓 DriveLM、BF16 LoRA、上下文和全局 batch 一致的训练吞吐；不采用量化聊天、API 输出速度或不同参数规模代替 |
| MobileNetV2 | [固定 ModelZoo 模型说明](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)，核对本地固定上游 Git 对象 | 原仓结果表为 NPU/O2 等配置，本次未找到可采用的同条件 GPU 基准；不能将原仓 NPU 数字标为 GPU 数据。保留待测 |

“未找到”限定于本次检查的来源，不表示网上不存在其他实验。以上参考不补进 NPU 实测 JSON，也不填成“GPU 实测已通过”；本项目的 GPU 速度比仍为空，待机器确认后按下述协议补测。

## 原生 GPU 对照如何验收

以下是**待执行的对照协议**，不是已经通过的测试。用户确认 GPU 机器、型号及可用卡后，每个模型独立留证；已有短计时结果保留为历史记录。

1. 固定原仓提交、输入形状、真实样本列表和顺序、权重摘要、优化器、训练/评测模式、batch 与梯度累积。使用原生 CUDA 入口作为基线，记录 NPU 适配差异；先核验两侧输出、loss 和有效梯度的可比性。
2. 分别报告相同 FP32/确定性条件，以及上游支持的混合精度条件。明确 TF32/HF32、AMP、损失缩放、算子后端和分布式规约；不同条件分表记录，不合并计算倍率。
3. 单计算设备与多计算设备分别测试，列明物理卡/芯片数量、型号、显存、驱动、CUDA/CANN、框架、容器 ID、CPU、数据盘、数据缓存与 dataloader workers。固定全局 batch 的比较和固定每 rank batch 的扩展测试分别报告。
4. 每种条件至少预热 50 更新；正式计时至少 60 秒并独立重复至少 3 次，保存同步计时边界、更新数、原始时间、全局吞吐、P50/P95、峰值显存及重复运行范围。首次编译/数据准备时间另行记录；Qwen 同时记录有效 token 数和梯度累积。
5. 只用同一条件的结果计算：`吞吐比 = NPU samples/s ÷ GPU samples/s`，`吞吐变化 = (吞吐比 − 1) × 100%`。端到端完整训练用时、达到同等质量的用时和能效另需相应实验；不能仅由短吞吐外推。

各模型的原有独立运行入口与数据门槛继续保留，见 [LSTM](../PyTorch/contrib/cv/others/LSTM-Argoverse/FULL_VALIDATION.md)、[STGCN](../PyTorch/contrib/cv/others/STGCN-Traffic/FULL_VALIDATION.md)、[MobileNetV2](../PyTorch/contrib/cv/classification/MobileNetV2/FULL_VALIDATION.md)、[ResWorld](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md)。原仓质量表、其他硬件/不同 batch 的社区吞吐和本仓 CPU/NPU 数值对照，都不能替代本次原生 GPU 性能实测。
