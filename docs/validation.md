# 模型验证结果

最近一轮实机记录：**2026-10-19**，重新执行 LSTM、STGCN、MobileNetV2 的全部验证数据，完成 Qwen 全部 DriveLM 数据检查及真实双卡公开入口预检，并完成 ResWorld 原 runner 的完整 mini 轮次、EMA 续训和 CPU 数据转换对照。完整训练、数值修复、续训和长稳的前轮证据为 2026-10-18；较早功能记录为 2026-10-12。各次执行日期单独记录。

本轮完整数据验收范围为八项；[WorldEngine AlgEngine](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine)由独立仓负责，本轮不测试。下文 WorldEngine 仅保留历史来源。

性能数据单列于[NPU 实测性能与 GPU 对照状态](performance.md)，包含原始 JSON、batch 和实际计时秒数。目前八项均缺同条件的原生 CUDA/GPU 性能基线，不能给出 GPU 快慢比例。

## 2026-10-19 新增全量复验

三项均在 ARM64 Atlas 800I A3 上新建输出目录、实际执行至退出 0；使用固定镜像 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。本轮重新评测已有检查点，未将前轮训练改写为当天新训练。

| 模型 | 本轮完整输入与输出 | 本轮实测指标 | 新执行记录与原始结果 |
| --- | --- | --- | --- |
| LSTM-Argoverse | 全部 39,472 val 轨迹，2,368,320 个坐标；CPU/NPU 超差 0，max=4.25055623e-6 | NPU ADE=1.7011764259 m；FDE=3.8735608773 m；MR=0.6511197811 | [时间、退出码、权重 SHA 与完整 JSON](../PyTorch/contrib/cv/others/LSTM-Argoverse/validation/full_revalidation_20261019.json) |
| STGCN-Traffic | 全部 val/test 各 1,885 窗口，共 859,560 个输出；CPU/NPU 超差 0，max=4.76837158e-6 | test MAE=2.3743455411；RMSE=4.1293943097；WMAPE=4.0883945979% | [时间、退出码、权重 SHA 与完整 JSON](../PyTorch/contrib/cv/others/STGCN-Traffic/validation/full_revalidation_20261019.json) |
| MobileNetV2 | CPU 与 NPU 分别处理全部 50,000 张 ImageNet val；逐个比较 50,000,000 logits，超差 0，max=6.86645508e-5 | Top-1=35,935/50,000=71.87%；Top-5=45,147/50,000=90.294% | [时间、退出码、权重 SHA 与完整 JSON](../PyTorch/contrib/cv/classification/MobileNetV2/validation/full_revalidation_20261019.json) |

LSTM 的 val 参与选模，STGCN 使用社区 70/15/15 协议，MobileNetV2 使用官方 V1 预训练权重。以上结果不表示隐藏排行榜、原论文全部协议或 MobileNetV2 从头 600 epoch 已验收。

## 2026-10-19 Qwen 完整数据与公开入口实测

固定镜像为 `local/llamafactory:a3-arm64-20261012`，image ID `sha256:99d33965ae4502b07bffe2c3248887795ae0672a0e79dd0aaa567d09cce811b5`。本轮使用 ARM64 Atlas 800I A3，逻辑设备 0、2 分属两张物理卡。

| 检查 | 实际结果 |
| --- | --- |
| 官方媒体 | 24,432 张训练图像、4,794 张官方验证图像；全部解码及文件摘要检查通过 |
| 有效问答 | 696 场景、4,072 帧；原始 377,956 条，明确排除 1 条空答案；626 / 70 场景划分为 340,119 train / 37,836 本地 dev |
| 原生上下文 | 全部 377,955 条通过原生 SFT 编码；1,574～2,629 token；4,096 上限可容纳全部输入和标签 |
| 官方验证问题 | 149 场景、799 帧、15,480 原始条目，保留并标记 1 条空问题；答案未公开 |
| 完整数据正向校验 | 官方归档、逐个解压图像、逐条转换问答、场景划分和上下文身份均通过，退出 0 |
| 公开双卡入口 | 4 场景 / 16 条真实问答、8 更新；训练及重载评测退出 0；504 个适配器张量、504 组优化器状态有限，252 个 LoRA B 全部非零 |
| 复现对照 | 三次独立输出目录（两次未复用、一次单帧复用）的适配器 SHA256 相同：`68da404a097ff580e6676478d912619c8e188ec7e3fc4924ec0ca5113db85874`；原双卡与公开单卡重载 loss 均为 1.8978482484817505 |
| 错误数据拦截 | Windows、ARM64 Linux 各 7 项通过；包括改名演示数据、任意目录、直接启动入口和结果指针不变检查 |
| 完整训练启动记录 | 全部 340,119 train / 37,836 dev 已完成原生预处理；loss@10=3.2349693298、loss@20=2.8300830841，保留原启动阶段数据 |
| 完整训练进度快照 | 2026-10-19 23:37 北京时间：2,111 / 63,774 更新（3.31%），第 2,110 步日志训练 loss=0.161；训练及完整留出评测尚未结束，见[带时间的原生进度](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/full_training_progress_20261019.json) |

[完整来源、摘要、命令、时间、退出码和当前后台阶段](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/drivelm_complete_data_20261019.json)；[独立模型指导](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md#完整数据训练drivelm-nuscenes-v11)。上述重载 loss 使用同一组 16 条预检训练样本，仅证明保存与重载功能。最终三轮训练和全部本地 dev 重载仍需对应终态结果；官方隐藏评分按 [DriveLM 官方提交流程](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/README.md#submit-to-test-server)办理。

## 完整数据准备与未完成项

2026-10-19 新增 ResWorld 原训练组件的四卡实测：GeoBEV 548 个兼容张量核对载入、86 个 SyncBN 模块每 rank 1,824 次调用；四 rank 各 8 更新、64 个不同真实 mini 训练帧；原 EMA 从 10,560 到 10,568，原 warmup / step 边界检查通过；参数摘要一致，模型、EMA、优化器有限，模型与 EMA 保存后重读逐位相同，退出 0。见[完整逐步数据与命令](../PyTorch/contrib/cv/others/ResWorld/validation/original_protocol_preflight_20261019.json)和[独立入口](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md#9-geobevsyncbnema-与-warmup-的四卡预检)。此项是 mini 训练组件预检；正式完整数据下 12 epoch 及任务精度尚未完成。

之后完成 **ResWorld 原 runner 完整 mini 轮次与 EMA 精确恢复**：323 条标注，四物理卡每卡 batch=2，41 更新/轮；一轮后重启续到两轮，对照独立连续两轮，2,762 个完整状态张量及四 rank 第二轮 164 条轨迹全部逐位相同。累计 82 更新、EMA=10,642，最终两份检查点 SHA256 同为 `6efe38550d7a3688fb75168990e5375da0b89ee1dfe69458ac2d5cd704748ac4`；公开 CPU 比较入口再次退出 0。实际脚本、时刻、逐步输入/loss/分桶及状态比较见[原始证据](../PyTorch/contrib/cv/others/ResWorld/validation/native_epoch_resume_20261019.json)，命令见[独立指导](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md#10-原-runner-的完整-mini-轮次与-ema-续训)。该次覆盖 workers=0、轮次边界；后续原 workers=8 复验见下文，正式 trainval/12 epoch 和任务精度继续单独验收。

同日另完成 **ResWorld 原配置 workers=8/rank** 的四卡恢复复验，实际观察四组各 8 个 DataLoader worker。三次训练和公开 CPU 比较全部退出 0；全部 323 条真实 mini 标注、41 更新/轮，一轮重启续到两轮与独立连续两轮的 2,762 个完整状态张量、四 rank 第二轮 164 条输入/loss/lr/EMA/分桶记录全部逐位相同。累计 82 更新、EMA=10,642，最终两份文件 SHA256 同为 `79cd050c8703998fcff41e3367fc2cca804d1407bd07ab59e2d4b2489786a055`。见[完整原始证据](../PyTorch/contrib/cv/others/ResWorld/validation/native_epoch_resume_workers8_20261019.json)及[独立命令](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md#原配置-workers8-的四卡恢复复验)。本次不改变完整数据与任务质量的未完成状态。

ResWorld 正式数据准备入口的三个负向用例通过；修复原模型算子导入及转换器 CLI 导入依赖后，**不映射 NPU 的公开 CPU 控制**重新生成全部 mini 标注/深度：14,140 个数组及其他标注字段、2,424 个深度文件与原结果一致。见[命令、来源、失败与回归结果](../PyTorch/contrib/cv/others/ResWorld/validation/full_preparation_guards_20261019.json)。正式 trainval 转换已排队等待全部传感器文件，正向完整转换尚未执行。10 月 18 日离线 ZIP 尚未更新这些新增入口。

BEVFormer、Sparse4D、Apollo、ResWorld 的正式 nuScenes trainval 尚在下载/准备，当前不填完整 mAP/NDS/mIoU/规划精度结果。官方 [AWS 公开数据登记](https://registry.opendata.aws/motional-nuscenes/)提供匿名下载；本次十个完整传感器 blobs、metadata、CAN、地图归档合计 316,066,113,900 bytes，另需 ResWorld 的 lidarseg 与投影深度。完成传输、原始校验和逐文件覆盖检查后才能开始相应完整数据验收。

Apollo 的完整 OCC 标签已于 2026-10-19 传输、校验及解压：700 train 场景 / 28,130 样本，150 val 场景 / 6,019 样本；全部 102,447 个标签文件的 NPY header 与数据长度检查通过，场景/样本与正式 nuScenes metadata 完全对应，train/val 无重叠。见[完整 OCC 数据检查](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/full_occ_data_20261019.json)。这只完成标签准备，相机/点云、CAN、派生 PKL 及模型完整训练/评测仍未完成。

Qwen3-VL-8B 已完成完整 DriveLM 文件、六相机媒体、场景划分、转换和全部样本上下文检查，真实双卡公开入口预检通过。新三轮训练流程已提交后台，完整训练与 37,836 条本地留出数据的最终重载尚未完成；官方隐藏答案未公开。旧 [full 入口负向回归](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/full_data_guard_20261019.json)为本轮较早阶段的修复记录，当前完整协议以新增证据为准。

MobileNetV2 的完整 ImageNet train 仍在续传，从头 600 epoch 未执行完。完整数据包、训练完成、任务精度与 CPU/NPU 一致性分别留证。

## 结果总表

| 模型与独立指导 | 实测日期 | 已取得的结果 | 范围与未完成项 |
| --- | --- | --- | --- |
| [BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2026-10-12 | nuScenes mini；双卡 162 更新、strict 重载及单帧有限性通过 | 全量 mAP/NDS、收敛、性能/续训/长稳未测 |
| [Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2026-10-12 | nuScenes mini；双卡 10 更新、strict 重载及单帧有限性通过 | 全量检测指标、收敛、性能/续训/长稳未测 |
| [Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2026-10-12 / 19 | 完整 DriveLM 数据和 377,955 条编码检查通过；两次真实数据双卡八步预检及重载通过；新三轮流程已提交后台 | 完整训练、完整本地 dev 重载与官方隐藏评分未完成 |
| [Apollo-Vision-Net](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 2026-10-12 | 官方 DLA 初始化；242 train/162 val；四卡 10 更新及全部 val 帧有限性通过 | 完整检测 mAP/占用 mIoU、收敛、性能/续训/长稳未测 |
| [LSTM-Argoverse](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 2026-10-18 训练 / 19 复验 | 全 train/val 三阶段 725 epoch / 292,175 更新；全输出、梯度、严格续训通过 | val 参与选模；独立排行榜/驾驶业务目标未验收 |
| [STGCN-Traffic](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 2026-10-18 训练 / 19 复验 | PeMSD7(M) 全量 52 epoch / 14,404 更新；独立 test、数值、续训、30 分钟通过 | 70/15/15 社区协议；原论文划分/METR-LA/其他 PeMS 未验收 |
| [ResWorld](../PyTorch/contrib/cv/others/ResWorld/README.md) | 2026-10-18 / 19 | 原数值修复、五份权重 446 梯度、30 分钟及原 65 帧对照；新增原 runner 两轮/EMA 精确恢复，2,762 状态张量与 164 条轨迹全同，CPU 数据转换对照通过 | 真实 mini、workers=0 和 8 各自的轮次控制；正式 trainval/12 epoch 及驾驶业务精度未完成 |
| [MobileNetV2](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 2026-10-18 与 19 | 官方 V1 全部 ImageNet 50,000 val；功能训练、数值、续训及 30 分钟通过 | 官方预训练评测；600 epoch 从头训练及驾驶业务未完成 |

## 数据与实测数值

### BEVFormer

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | nuScenes mini：323 train / 81 val；CAN 扩展缺失的 404 条记录使用零填充 |
| 训练配置 | 2 张物理卡；FP16 混合精度（loss_scale=512）；全局 batch=2；1 epoch / 162 次更新 |
| 保存状态 | 1,089 个模型张量、557 组优化器状态有限；优化器 step=162 |
| 独立验证 | strict 重载；1 帧真实输入推理输出有限 |
| 未完成 | 完整 nuScenes mAP/NDS、收敛、吞吐、续训及长稳；单帧有限性不等于检测精度 |

[实际结果 JSON](../PyTorch/contrib/cv/detection/BEVFormer/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/cv/detection/BEVFormer/README.md)。

### Sparse4D

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | nuScenes mini 功能子集，非完整 trainval |
| 训练配置 | 2 张物理卡；FP16 混合精度（loss_scale=32）；全局 batch=2；10 次更新 |
| 保存状态 | 819 个模型张量、652 组优化器状态有限；优化器 step=10 |
| 独立验证 | strict 重载；1 帧真实输入推理输出有限 |
| 未完成 | 完整 nuScenes 检测指标、收敛、吞吐、续训及长稳；十步不能证明收敛 |

[实际结果 JSON](../PyTorch/contrib/cv/detection/Sparse4D/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/cv/detection/Sparse4D/README.md)。

### Qwen3-VL-8B（2026-10-12 历史演示）

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | 完整官方 8B 基座；mllm_demo 6 条演示图文；另用独立合成图片检查内容 |
| 训练配置 | 2 张物理卡；BF16 LoRA SFT；10 次更新；20261012 固定镜像 |
| 保存状态 | 504 个适配器张量有限；252/252 个 LoRA B 非零；504 组优化器状态 step=10；2 rank 产物 |
| loss | 首条 3.2745351791，末条 2.5068483353；仅为十步记录，非收敛验收 |
| 独立验证 | 重新加载适配器，正确描述独立图片中的红色正方形；检查 1 张合成图 |
| 未完成 | 真实业务图文任务精度、完整训练、吞吐、续训及长稳 |

[实际结果 JSON](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md)。

### WorldEngine（历史，本轮不验收）

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | 官方 NAVTEST 3 段日志 / 140 帧 / 64 时序锚点；真实图像、轨迹及 PDM 缓存 |
| 训练配置 | AlgEngine VADv2 imitation learning；官方感知初始化；4 张物理卡；FP32；10 次更新 |
| 保存状态 | 889 个模型张量、259 组优化器状态有限；优化器 step=10 |
| 独立验证 | 4 rank 检查通过；保存的模型/优化器及逐 rank 计算结果有限 |
| 未完成 | NAVTEST 复用为功能训练，独立验证锚点为 0；完整 navtrain、PDM 精度、SimEngine、RL、吞吐、续训及长稳 |

[实际结果 JSON](../PyTorch/contrib/cv/others/WorldEngine/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/cv/others/WorldEngine/README.md)。

### Apollo-Vision-Net

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | 官方划分与 nuScenes mini 交集：242 train / 162 val，训练验证不相交；完整 OCC 归档结构已校验 |
| 训练配置 | 官方 DLA 深度预训练初始化；4 张物理卡；FP32；10 次更新 |
| 保存状态 | 600 个模型张量、465 组优化器状态有限；优化器 step=10 |
| 独立验证 | strict 重载；全部 162 个独立验证帧的原始 occupancy head 和解码结果有限 |
| 未完成 | 完整 nuScenes 检测 mAP / 占用 mIoU、收敛、吞吐、续训及长稳；有限值检查不等于任务精度 |

[实际结果 JSON](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md)。

### LSTM-Argoverse

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据与协议 | Argoverse v1.1：205,942 train / 39,472 val；20 帧观察、30 帧预测；原 1/10/30 curriculum |
| 完整训练 | 2 张物理卡；全局 batch=512；725 epoch / 292,175 次更新；最佳 epoch=535 |
| 轨迹指标 | 全 val：ADE=1.70117643 m；FDE=3.87356088 m；MR=0.6511197811（2 m、K=1） |
| CPU/NPU 全量输出 | 2,368,320 个归一化坐标，超差 0；最大绝对误差 4.25055623e-06；rtol=atol=0.003 |
| 梯度 | 14/14 个有效梯度通过；整体相对 L2=1.54778409e-06，上限 0.01 |
| 严格续训 | 早期 60/60、成熟 curriculum 102/102 完整状态张量逐位相同；最大误差 0 |
| 持续训练 | 完整 curriculum 累计 epoch 训练用时 10449.386 s；独立检查点重载通过 |
| 未完成 | val 参与选模，不能称独立排行榜 test；驾驶业务精度及生产性能目标未验收 |

[实际结果 JSON](../PyTorch/contrib/cv/others/LSTM-Argoverse/validation/extended_acceptance_20261018.json)；[复现指导](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md)；[完整训练/验证步骤及门槛](../PyTorch/contrib/cv/others/LSTM-Argoverse/FULL_VALIDATION.md)。

### STGCN-Traffic

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据与协议 | PeMSD7(M)，228 传感器；70/15/15 时间划分；8,857 train / 1,885 val / 1,885 test 窗口；预测 15 min |
| 完整训练 | 2 张物理卡；全局 batch=32；52 epoch / 14,404 次更新；最佳 epoch=42；验证早停 |
| 独立 test | MAE=2.3743455411；RMSE=4.1293943097；WMAPE=4.0883945979%；沿用原始速度数值单位 |
| CPU/NPU 全量输出 | 859,560 个 val/test 输出，超差 0；归一化最大误差 val=4.529953e-6 / test=4.768372e-6；rtol=atol=0.003 |
| 梯度 | 28/28 个有效梯度通过；整体相对 L2=8.6354638e-7，上限 0.01 |
| 严格续训 | 164/164 完整状态张量逐位相同；最大误差 0 |
| 持续训练 | 双卡 1,800.139611 s / 新增 86,190 次更新；保存状态及独立重载通过 |
| 未完成 | 这是所选社区 PyTorch 协议；原论文 60/20/20、METR-LA 和其他 PeMS 未验收 |

[实际结果 JSON](../PyTorch/contrib/cv/others/STGCN-Traffic/validation/extended_acceptance_20261018.json)；[复现指导](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md)；[完整训练/验证步骤及门槛](../PyTorch/contrib/cv/others/STGCN-Traffic/FULL_VALIDATION.md)。

### ResWorld

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据与协议 | 真实 nuScenes mini：323 train / 81 val 标注帧，8/2 场景；CAN 20 文件、深度 2,424 文件 |
| 功能训练 | 4 张物理卡；FP32；真实增强数据 20 步控制，之后另做持续训练；不是原 12 epoch 全量训练 |
| 五份权重数值回归 | 原失败、后续失败、旧长稳、新 20 步、最终长稳权重各 446/446 有效梯度通过；原门槛保持 |
| 最终权重梯度 | 整体相对 L2=2.83294570892e-5，上限 0.01；逐参数、loss 及一步参数门槛均通过 |
| CPU 参考检查 | 常量几何投影修复前后 446/446 个 CPU 参考梯度逐位相同；时序等价化范围另见数值说明 |
| 严格续训 | 四卡 10+10 对连续 20：2,054/2,054 完整状态张量逐位相同，max=0；四 rank 增强输入 SHA 匹配 |
| 持续训练及重载 | 四卡 1,811.967955 s / 新增 660 次更新；最终累计 step=680；独立重载 CPU/NPU max=3.8146973e-6 |
| 作者权重评测 | 原规划/碰撞函数全部 65 帧 CPU/NPU 对照通过；轨迹 rtol=atol=0.003，逐帧离散碰撞完全一致 |
| 未完成 | 正式 nuScenes trainval、原 12 epoch / EMA / SyncBN / warmup；作者权重可能见过 mini，不作为独立驾驶精度验收 |

[实际结果 JSON](../PyTorch/contrib/cv/others/ResWorld/validation/extended_acceptance_20261018.json)；[复现指导](../PyTorch/contrib/cv/others/ResWorld/README.md)；[完整训练/验证步骤及门槛](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md)。

### MobileNetV2

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 功能训练数据 | Imagenette：100 train / 40 val；保留 1,000 类输出头；2 张物理卡、全局 batch=8、20 步控制 |
| 正式验证数据 | ImageNet-2012 全部 50,000 val；官方 V1 预训练权重，非本项目从头训练权重 |
| 全量分类指标 | CPU 与 NPU：Top-1=35,935/50,000=71.87%；Top-5=45,147/50,000=90.294% |
| CPU/NPU 全量输出 | 50,000,000 个 logits，超差 0；最大绝对误差 6.8664551e-5；完整判据见全量 JSON |
| 梯度 | 固定源权重、新 20 步权重各 158/158 有效梯度通过；整体相对 L2 分别 1.1653562e-4 / 4.0335445e-4，上限 0.01 |
| 严格续训 | 478/478 完整状态张量逐位相同；最大误差 0；真实增强后输入 SHA 一致 |
| 持续训练 | 双卡 1,800.501374 s / 新增 33,570 次更新；独立重载 CPU/NPU max=2.4795532e-5 |
| 未完成 | 完整 ImageNet 600 epoch 从头训练；驾驶检测、分割和规划业务指标 |

[实际结果 JSON](../PyTorch/contrib/cv/classification/MobileNetV2/validation/extended_acceptance_20261018.json)；[复现指导](../PyTorch/contrib/cv/classification/MobileNetV2/README.md)；[完整训练/验证步骤及门槛](../PyTorch/contrib/cv/classification/MobileNetV2/FULL_VALIDATION.md)。

## 独立部署与证据入口

| 模型 | 独立训练与验证 | 目录独立性 | 镜像 | 数据范围 |
| --- | --- | --- | --- | --- |
| BEVFormer | [执行命令/时间/退出码](../PyTorch/contrib/cv/detection/BEVFormer/validation/entry_acceptance.json) | [换目录复验](../PyTorch/contrib/cv/detection/BEVFormer/validation/path_relocation.json) | [单镜像导入](../PyTorch/contrib/cv/detection/BEVFormer/validation/single_image_import.json) | [完整数据门禁](../PyTorch/contrib/cv/detection/BEVFormer/validation/full_data_guard.json) |
| Sparse4D | [执行命令/时间/退出码](../PyTorch/contrib/cv/detection/Sparse4D/validation/entry_acceptance.json) | [换目录复验](../PyTorch/contrib/cv/detection/Sparse4D/validation/path_relocation.json) | [单镜像导入](../PyTorch/contrib/cv/detection/Sparse4D/validation/single_image_import.json) | [完整数据门禁](../PyTorch/contrib/cv/detection/Sparse4D/validation/full_data_guard.json) |
| Qwen3-VL-8B | [执行命令/时间/退出码](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/entry_acceptance.json) | [换目录复验](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/path_relocation.json) | [单镜像导入](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/single_image_import.json) | [完整数据门禁](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/full_data_guard.json) |
| WorldEngine | [执行命令/时间/退出码](../PyTorch/contrib/cv/others/WorldEngine/validation/entry_acceptance.json) | [换目录复验](../PyTorch/contrib/cv/others/WorldEngine/validation/path_relocation.json) | [单镜像导入](../PyTorch/contrib/cv/others/WorldEngine/validation/single_image_import.json) | [完整数据门禁](../PyTorch/contrib/cv/others/WorldEngine/validation/full_data_guard.json) |
| Apollo-Vision-Net | [执行命令/时间/退出码](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/entry_acceptance.json) | [换目录复验](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/path_relocation.json) | [单镜像导入](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/single_image_import.json) | [完整数据门禁](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/full_data_guard.json) |

后四项的完整执行阶段、退出码、检查点 SHA、梯度清单、吞吐及持续训练均收录在各自 `validation/extended_acceptance_20261018.json`，从各模型 README / FULL_VALIDATION.md 进入。当前结果保留原测试门槛；失败回归证据按历史身份保存，不能当作当前通过结果，也不能把修复后的通过记录改写成此前已通过。

## 环境及结论的适用范围

实测硬件为 ARM64 Atlas 800I A3 推理系列（板卡 `IT22HMDA_4_S`，8 张双芯片物理卡 / 16 个逻辑设备）。两卡/四卡测试每个 rank 使用不同物理卡的一个逻辑芯片。后四项扩展记录来自用户提供的第二台机器；原五项记录来自第一台机器。具体测试环境与 image ID 见[环境与镜像](environment.md)及每个模型的 `images.lock.json`。

Atlas 800I A3 上的实际实验不改变华为官方训练硬件支持范围。功能/数值通过、官方预训练评测、完整参考协议训练和驾驶业务验收分别按上表判断；未完成项没有对应的通过结论。新机器按各模型 README 实际重跑，不能仅查看本页替代验证。
