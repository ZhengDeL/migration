# 模型验证结果

最近一轮实机记录：**2026-10-20**：BEVFormer、Apollo 和 Sparse4D 的新 mini 训练及全部 mini 留出集评测已完成；BEVFormer/Sparse4D 作者权重对照保留。早期短训检查点的低指标、失败及 CPU 控制保留；Qwen 全量训练继续。2026-10-19 的三项全量复验、ResWorld 原 runner/EMA/CPU 数据对照和更早结果保留实际执行日期。

本轮完整数据验收范围为八项；[WorldEngine AlgEngine](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine)由独立仓负责，本轮不测试。下文 WorldEngine 仅保留历史来源。

性能数据单列于[NPU 实测性能与 GPU 对照状态](performance.md)，包含原始 JSON、batch 和实际计时秒数。目前八项均缺同条件的原生 CUDA/GPU 性能基线，不能给出 GPU 快慢比例。

## 2026-10-20 后续修复与重新训练

截至 **12:13 北京时间**核对，三项新 mini 训练与末尾全部留出帧评测均退出 0。BEVFormer 和 Apollo 另对最终检查点摘要及全部逐 rank JSONL 连续更新、有限值完成只读复核；原失败/低指标保留。以下是 mini 流程与数值结果，**三项正式 trainval 任务质量均未验收**。

| 模型 | 新训练与全部 mini 留出评测 | 验收边界与证据 |
| --- | --- | --- |
| BEVFormer | 原 FCOS3D 初始化、原视频 runner；双卡各 3,888 更新 / 24 epoch；81/81 帧，mAP=0.0554680385、NDS=0.0994282077 | [新训练命令、逐 rank/逐帧、全部指标及检查点 SHA](../PyTorch/contrib/cv/detection/BEVFormer/validation/mini_training_evaluation_20261020.json)；作者全量权重控制为 mAP=0.3623209477、NDS=0.4103581189，[控制证据](../PyTorch/contrib/cv/detection/BEVFormer/validation/official_checkpoint_mini_evaluation_20261020.json)；两种权重不能混作同一结果 |
| Apollo VisionNet | 官方 DLA 骨干初始化；四卡各 1,464 更新 / 24 epoch；162/162 OCC 帧，IoU=18.6466263092%、mIoU=4.7205596444% | [四 rank、逐帧、原混淆矩阵与检查点 SHA](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/mini_training_evaluation_20261020.json)；修复 spawn 后完成全日程，新权重 OCC 指标非零，仍未完成正式 trainval 精度验收 |
| Sparse4D | ImageNet 骨干初始化；双卡各 2,600 更新；自动评测及另一次公开单卡复测各 81/81 帧，mAP=0.0604165495、NDS=0.1097741302 相同 | [训练、检查点 SHA 与两次完整指标](../PyTorch/contrib/cv/detection/Sparse4D/validation/mini_training_evaluation_20261020.json)；新检查点未复现整套推理溢出，旧十步权重失败保留，正式 trainval 精度未验收 |

BEVFormer 的原转换器 CPU 入口另已实跑：全部 404 帧、17,980 个数组、123,038 个标量对照无差异，mini 冒充完整数据被拒绝且准备目录/结果指针不变。[实际命令、字段比较、源文件摘要与正式训练等待状态](../PyTorch/contrib/cv/detection/BEVFormer/validation/original_converter_control_20261020.json)。正式 trainval 转换/训练队列已等待原始数据全覆盖核验；尚未执行完整转换或完整训练。

新入口为 Apollo 自己的 `test/train_mini_4p.sh`，以及 BEVFormer、Sparse4D 各自的 `test/train_mini_2p.sh`；三项均已自动重载、完整 mini 留出评测并保存模型级结果；Sparse4D 另完成公开单卡入口复测。mini 训练用于进一步定位和修复当前质量问题，正式 trainval 训练仍需完整数据；三项均未标为正式质量验收通过。10 月 18 日旧网盘 ZIP 尚未更新。


**同轮完整任务进度（12:14 北京时间）**：Qwen3-VL-8B 为 6,281 / 63,774 更新，训练继续，尚无最终 dev 指标，[原始进度](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/full_training_progress_20261020.json)。正式 nuScenes 14 个归档中 8 个已完成当前官方对象完整性核验，正在下载第 05 个 blobs 分包；完整原始文件全覆盖、BEVFormer/ResWorld 正式数据队列仍等待，不能用 mini 代替。

### 13:09 北京时间追加：精度问题仍未关闭

BEVFormer、Apollo 新检查点各通过公开单卡入口在新目录复测全部 81 / 162 帧，指标与训练末尾自动评测完全相同。Apollo 另以真实训练输入核验适配过的 focal loss：检测 900×10、OCC 640,000×16 的原 CPU 公式/NPU loss 与梯度对照通过，未修改原 10.0 OCC loss 权重、背景标签规则、平均因子或评测阈值。[真实输入摘要、预设容差、逐项差值、命令与退出码](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/focal_loss_precision_20261020.json)。此证据只覆盖上述两个 loss 输入，不能代替完整网络/CUDA 对照。

当前低 mini 分数的来源尚未全部定位。完整数据、CAN/时序和标签、适配算子、原训练日程/收敛需分别验证；作者全量权重控制与 mini 训练权重不构成同条件对照，不能仅据两者分差判定迁移损失或训练通过。三项的正式质量结论继续为未验收。

## 2026-10-20 早期短训检查点评测与剩余验收

在同一固定 DrivingSDK 镜像、三个独立模型目录执行；逻辑卡 4 / 6 / 8 分属物理卡 2 / 3 / 4。Qwen 保留物理卡 0、1 继续训练。全部 31,223 个 mini 原始文件逐份 SHA256 通过。表中“流程完成”仅指加载、全部推理和原仓指标计算已完成；短训权重的任务精度尚不能验收。

| 模型与权重 | 本轮实际数据 / 指标 | 结果与完整证据 |
| --- | --- | --- |
| BEVFormer，既有 162 更新检查点 | 81/81 标准 mini_val 帧；mAP=0.00003540314127，NDS=0.0178060191423（0～1） | 原仓评测流程完成、退出 0；质量未验收；[逐帧/原始指标/镜像/命令/退出码](../PyTorch/contrib/cv/detection/BEVFormer/validation/complete_mini_evaluation_20261020.json) |
| Apollo VisionNet，既有 10 更新检查点 | 162/162 原 OCC heldout 帧；IoU=0%、mIoU=0%，precision/recall=0；404 GT + 404 原始可见性 mask 校验 | 原仓 OCC 评测流程完成、退出 0；质量未验收；[混淆矩阵/逐帧/原始指标与 mask 修复复验](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/complete_mini_evaluation_20261020.json) |
| Sparse4D，既有 10 更新检查点 | 在第 36 帧失败；FP16/FP32 均复现。log 尺寸 89.4342651367 超过 FP32 exp 上限 88.7228393555，CPU 同输入同样溢出 | 完整 mini 未通过、退出 1；未填写 mAP/NDS、未裁剪/跳帧；[失败帧、模块诊断、FP16/FP32/CPU 控制](../PyTorch/contrib/cv/detection/Sparse4D/validation/complete_mini_failure_20261020.json) |
| Sparse4D，作者已训练 v3 R50 权重 | 81/81 标准 mini_val 帧有限；mAP=0.4635453126、NDS=0.5071118864（0～1）；显式 profile 与默认公开入口指标相同 | 两次独立输出均退出 0；[原始结果、两次命令和权重/代码 SHA](../PyTorch/contrib/cv/detection/Sparse4D/validation/official_checkpoint_mini_evaluation_20261020.json)；本项目完整 trainval 训练仍未完成 |

Apollo 修复了本项目准备脚本遗漏的 `occ_invalid_path`，原 mask 字节、可见性规则、0.25 阈值和评测公式保持；原 split 仍是 242 train / 162 val。原 nuScenes detection 入口要求标准 mini_val 81 帧，因此本次仅计算匹配原 OCC split 的占用指标。BEVFormer 的五项按 nuScenes 定义不适用的 cone/barrier 误差字段编码为 null，所有网络预测和 mAP/NDS 仍严格要求有限。

### 当前尚需完成

| 模型 | 后续必需验证 | 当前条件 |
| --- | --- | --- |
| BEVFormer | 正式 trainval 训练、收敛、完整 mAP/NDS；完整训练的续训/稳定性 | 作者权重控制及新 24 轮 mini 训练均完成 81 帧；正式 nuScenes 分包继续下载 |
| Sparse4D | 正式 trainval 训练、收敛与完整检测指标；完整训练检查点数值/续训/稳定性 | 作者权重及新 2,600 更新检查点均完成 81 帧；旧十步权重失败保留；正式 nuScenes 继续下载 |
| Qwen3-VL-8B | 三 epoch 完整训练、37,836 条本地 dev 重载评测、完整训练检查点/续训/稳定性 | 340,119 条 train / 37,836 dev 已齐备，双卡训练运行中；官方隐藏答案仍未公开 |
| Apollo VisionNet | 正式 trainval 完整训练、检测/OCC 指标与完整训练恢复/稳定性 | 新四卡 24 轮 mini 训练与 162 帧评测已完成，mIoU=4.72056%；正式图像/点云下载中，正式质量未验收 |
| LSTM-Argoverse | 所选协议已完成；业务/隐藏排行榜另行验收 | 全部数据、数值、续训和持续训练证据已完成 |
| STGCN-Traffic | 所选 PeMSD7(M) 70/15/15 协议已完成；其他数据/原论文协议另行验收 | 全量 test、数值、续训和持续训练证据已完成 |
| ResWorld | 正式 trainval / 原 12 epoch、完整驾驶任务指标及完整数据恢复/稳定性 | mini 数值/原 runner 恢复已完成；正式数据准备队列等待原始数据全部到齐 |
| MobileNetV2 | 完整 ImageNet 从头训练及其收敛/任务指标 | 50,000 val 预训练评测/数值已完成；ImageNet train 的 TLS 中断已启动分块重验/续传恢复 |

八模型仍无同条件 CUDA 机器实测。官方网上数据仅作来源明确的参考，不能换算 GPU/NPU 速度倍率。当前网盘九个旧 ZIP 未重新打包，尚不包含本轮新增入口/修复/证据；完整验收后按每模型独立归档。

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

MobileNetV2 的完整 ImageNet train 下载因 TLS EOF 中断，已下载的 115,720,847,360 / 147,897,477,120 bytes 保留。本轮已修复可重试传输错误导致全局退出的问题，以独占后台进程重验旧分块后续传；最终官方 MD5、传到服务器、数据展开和从头 600 epoch 仍未完成。完整数据包、训练完成、任务精度与 CPU/NPU 一致性分别留证。

## 结果总表

| 模型与独立指导 | 实测日期 | 已取得的结果 | 范围与未完成项 |
| --- | --- | --- | --- |
| [BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2026-10-12 / 20 | 原 FCOS3D 初始化双卡 24 轮 / 3,888 更新完成，81 帧 mAP=0.055468 / NDS=0.099428；作者权重控制为 mAP=0.362321 / NDS=0.410358，见页首两份证据 | 新 mini 训练/数值流程完成，低指标仍不足以验收正式质量；完整 trainval 训练未完成 |
| [Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2026-10-12 / 20 | 新原 ImageNet 骨干双卡 2,600 更新完成，自动和公开单卡各 81 帧不溢出，mAP=0.060417 / NDS=0.109774 相同；作者权重对照另有独立结果 | 旧十步失败保留；新 mini 指标较低，正式 trainval 完整训练/任务质量及 GPU 对齐未验收 |
| [Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2026-10-12 / 19 | 完整 DriveLM 数据和 377,955 条编码检查通过；两次真实数据双卡八步预检及重载通过；新三轮流程已提交后台 | 完整训练、完整本地 dev 重载与官方隐藏评分未完成 |
| [Apollo-Vision-Net](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 2026-10-12 / 20 | mask/spawn 修复后，从原 DLA 骨干四卡完成 24 轮 / 1,464 更新，162 帧 IoU=18.646626% / mIoU=4.720560%；旧十步权重零指标保留 | 新 mini 训练/数值流程完成；正式 trainval 检测/OCC、收敛、恢复/稳定性未验收 |
| [LSTM-Argoverse](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 2026-10-18 训练 / 19 复验 | 全 train/val 三阶段 725 epoch / 292,175 更新；全输出、梯度、严格续训通过 | val 参与选模；独立排行榜/驾驶业务目标未验收 |
| [STGCN-Traffic](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 2026-10-18 训练 / 19 复验 | PeMSD7(M) 全量 52 epoch / 14,404 更新；独立 test、数值、续训、30 分钟通过 | 70/15/15 社区协议；原论文划分/METR-LA/其他 PeMS 未验收 |
| [ResWorld](../PyTorch/contrib/cv/others/ResWorld/README.md) | 2026-10-18 / 19 | 原数值修复、五份权重 446 梯度、30 分钟及原 65 帧对照；新增原 runner 两轮/EMA 精确恢复，2,762 状态张量与 164 条轨迹全同，CPU 数据转换对照通过 | 真实 mini、workers=0 和 8 各自的轮次控制；正式 trainval/12 epoch 及驾驶业务精度未完成 |
| [MobileNetV2](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 2026-10-18 与 19 | 官方 V1 全部 ImageNet 50,000 val；功能训练、数值、续训及 30 分钟通过 | 官方预训练评测；600 epoch 从头训练及驾驶业务未完成 |

## 数据与实测数值

### BEVFormer

以下为 10 月 12 日功能短训历史；10 月 20 日作者权重评测和新增训练进度见页首及模型独立指导，不以此表代表最新任务质量。

| 检查项目 | 本项目实测数据 |
| --- | --- |
| 数据范围 | nuScenes mini：323 train / 81 val；CAN 扩展缺失的 404 条记录使用零填充 |
| 训练配置 | 2 张物理卡；FP16 混合精度（loss_scale=512）；全局 batch=2；1 epoch / 162 次更新 |
| 保存状态 | 1,089 个模型张量、557 组优化器状态有限；优化器 step=162 |
| 独立验证 | strict 重载；1 帧真实输入推理输出有限 |
| 未完成 | 完整 nuScenes mAP/NDS、收敛、吞吐、续训及长稳；单帧有限性不等于检测精度 |

[实际结果 JSON](../PyTorch/contrib/cv/detection/BEVFormer/validation/reference_run/validation_result.json)；[复现指导](../PyTorch/contrib/cv/detection/BEVFormer/README.md)。

### Sparse4D

以下为 10 月 12 日十步功能短训历史；10 月 20 日旧权重失败、作者权重控制及新 2,600 更新终态见页首和模型独立 JSON。

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

以下为 10 月 12 日十步功能短训历史；10 月 20 日旧权重完整 OCC 指标和新增四卡训练进度见页首，不将有限性通过作为 mIoU 验收。

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
