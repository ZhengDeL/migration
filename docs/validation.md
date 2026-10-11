# 1# 实测结果（2026-10-11）

1# 为 Atlas 800I A3，8 张双芯片物理卡。两卡 / 四卡试验选择不同物理卡的一个逻辑芯片，每卡一个 rank；四卡结果也来自 1#。当前属于实验性功能验证，官方训练硬件支持边界见[环境指南](environment.md)。

## 当前模型结果

| 模型 | 输入和运行范围 | 实际检查结果 | 未覆盖内容 |
| --- | --- | --- | --- |
| BEVFormer | 2 物理卡、nuScenes mini 323 train / 81 val，1 epoch / 162 迭代 | loss 有限，1089 模型张量 / 557 优化器状态可读取 | CAN bus 扩展缺失、404 条记录填零；完整 mAP/NDS、性能和稳定性未测 |
| Sparse4D | 2 物理卡、nuScenes mini，10 迭代 | loss 有限，819 模型张量 / 652 优化器状态可读取；上游 HEAD 已更新，运行代码与已训练版本一致 | 首末 loss 37.1234 / 41.4400，不能称收敛；完整指标和普通 AdamW 性能未测 |
| Qwen3-VL-8B | 新 20261011 A3 镜像、2 物理卡、完整官方 8B 权重、6 条演示图文，BF16 LoRA 10 步 | 504 适配器张量有限、252 LoRA B 非零、优化器 step=10、两 rank RNG 保存；重新加载后正确识别独立红色正方形图片 | 一张合成图仅作内容检查；业务精度、性能和稳定性未测 |
| WorldEngine | 4 物理卡、官方 NAVTEST 三段日志 / 140 帧 / 64 时序锚点；真实图像及 PDM 缓存、官方感知基座，FP32 10 步 | 四 rank 原始输出、掩码处理前 loss、每步各 259 梯度张量有限；被检查参数更新并跨 rank 一致；889 模型张量 / 259 优化器状态有限，step=10 | NAVTEST 复用为功能训练；完整 navtrain、独立 PDM 指标、SimEngine、强化学习未测 |
| Apollo | 完整 OCC 归档校验；官方 train/val 与 mini 交集 242 train / 162 独立 val；官方 DLA 深度初始化，4 物理卡 FP32 10 步 | 初始化 222/222 张量完全匹配；四 rank 原始输出、掩码前 loss、每步各 465 梯度张量有限；被检查参数更新并跨 rank 一致；600 模型张量 / 465 优化器状态有限，step=10；严格重载后 162/162 独立验证帧输出有限 | 未计算检测 mAP / 占用 mIoU，未作完整 nuScenes 训练、性能、长稳或继续训练验收 |

公开实机证据：[Apollo 最新 rank 0](../verification/host1_20261011/apollo_official_init/rank0.json)、[rank 1](../verification/host1_20261011/apollo_official_init/rank1.json)、[rank 2](../verification/host1_20261011/apollo_official_init/rank2.json)、[rank 3](../verification/host1_20261011/apollo_official_init/rank3.json)、[全部验证帧](../verification/host1_20261011/apollo_official_init/heldout_check.json)；[Qwen 新镜像保存检查](../verification/host1_20261011/qwen_latest/checkpoint_check.json)、[图文推理](../verification/host1_20261011/qwen_latest/inference_check.json)；[WorldEngine rank 0](../verification/host1_20261011/worldengine_real/rank0.json)及同目录各 rank、[run](../verification/host1_20261011/worldengine_real/run.json)。原始日志保留在交付报告中。

训练发生在最终说明文档冻结之前，当前结果的 [Apollo run](../verification/host1_20261011/apollo_official_init/run.json) / [Qwen run](../verification/host1_20261011/qwen_latest/run.json)标注实际镜像及版本范围。最终归档另用独立目录恢复验证，报告随交付保存；不能把之后的文档提交号当作之前训练使用的代码提交号。

## 数据、初始化与适配依据

Apollo 完整原归档为 15,840,721,593 字节，观测 SHA256 `52ae4e3637fc4c1b1aa2d1fb4273676c06606200566f3d436e2a359edd1b1282`。本地与服务器 SHA256 一致，gzip CRC 与 103302 个 tar 成员全部检查；没有取得发布方公布的哈希。普通文件 payload 共 78,287,038,688 字节。全归档结构校验不等于全 NPY 语义检查：本次逐数组验证 404 个 mini 标签，原标签复制前后 SHA256 一致、官方 train/val 不相交、CAN bus 全部非零。[数据 manifest](../verification/host1_20261011/apollo_official_init/data_manifest.json)记录具体范围。

**更正此前初始化说明：** 官方固定源码实际已包含所需 DLA 权重，之前“未提供”的描述有误。其 Git blob `af42f8693fd64f47021e64d6b530331cf075e901`、大小和 SHA256 已核验，模型训练前全部 222 个初始化张量完全匹配。本表 Apollo 已改为官方初始化重跑结果；历史随机初始化 / VAL 复用结果仍保留，不作为当前独立验证证明。

WorldEngine 四官方功能资产共 5,589,709,908 字节，另有 3,932,288 字节词表；均通过发布方 SHA256。保留历史 / 未来连续帧和原始轨迹标签，只筛选样本并重写图像路径。[数据 manifest](../verification/host1_20261011/worldengine_real/data_manifest.json)为 64 训练锚点、零独立验证；官方基座 backbone / encoder 抽查张量匹配。范围为 AlgEngine imitation learning。

Apollo 编译 FocalLoss 的 `PyThreadState_Get` 崩溃由兼容补丁改用 MMDetection 已有的 PyTorch 公式；保留损失参数和注册映射。FP32 CPU/NPU 前后向 8 个案例通过 `atol=1e-5, rtol=1e-4`，最大梯度绝对误差约 6.20e-6，sum loss 绝对误差 0.001953125 满足相对误差标准。[对比证据](../verification/host1_20261011/apollo_real/focal_parity.json)只覆盖这些案例，未替代 CUDA 或完整精度回归。

## 当前保存产物

| 模型 | 文件 | 字节数 | SHA256 |
| --- | --- | ---: | --- |
| BEVFormer | `outputs/bevformer/epoch_1.pth` | 827201103 | `d50e854d079d28e1c09ef9c57d4959236ca7b9252198da971b24fa174544d70d` |
| Sparse4D | `outputs/sparse4d/iter_10.pth` | 577168640 | `acf68ff01ce0f865532f8ffd55df44490185bdcf2e733d95365402317ea4167d` |
| Qwen | `outputs/qwen3vl8b/adapter_model.safetensors` | 87368144 | `7ee34281193045e042789faa20a570fb6cf3ddc16df1c8aee0923c5640839c7e` |
| WorldEngine | `outputs/worldengine/mini/iter_10.pth` | 434886101 | `4ed2a4ca77ce9252964292faa604154a87a49d21197bf5fd05727caabff83ad8` |
| Apollo | `outputs/apollo_visionnet/mini/iter_10.pth` | 844544332 | `f2b2c77e1b6df0a58ffcafff308b18a0fa5774d67dbd8efa7d284229569d4a29` |

WorldEngine / Apollo 保存文件含优化器，已在匹配的 NPU 环境检查有限值和 step=10；SDK `meta.iter=9` 从零计数。尚未验证继续训练。权重本体不放 GitHub，包含在交付归档中。

## 历史证据

旧 [Apollo VAL 复用 / 随机初始化](../verification/host1_20261011/apollo_real/run.json)保存 SHA256 为 `5288adfd7555740da0a41ba04fd2bad30d866a6b9e628fe948d4bf628b576c3d`；完整划分随机初始化的中间结果 SHA256 为 `847690daa12b89924a917f5c26573affb8934a75a263415c57f73b640ec42731`。分别归档到 `mini_val_reuse_20261011` / `mini_random_full_split_20261011`。旧 Qwen 镜像与输出归档到回退标签和 `qwen3vl8b_baseline_20261011`。

低分辨率 [WorldEngine 合成检查](../verification/host1_20261011/worldengine/rank0.json) / [Apollo 合成检查](../verification/host1_20261011/apollo/rank0.json)仅是历史计算功能证据。Apollo 历史合成检查没有覆盖后续修复的编译 FocalLoss 路径。

## 镜像与基础计算

当前镜像 ID 见 [images.lock.json](../images.lock.json)，完整来源和实测版本见[环境指南](environment.md)，当日更新差异见[版本核验](latest_versions.md)。

DrivingSDK 两 rank / 四 rank 基础前后向、参数更新和 HCCL all-reduce 通过，sum 分别为 3 / 10。多尺度变形注意力前向及 value / location / attention 梯度与参考实现 allclose，最大绝对误差分别为 1.192e-6 / 2.384e-7 / 1.144e-5 / 2.861e-6，只覆盖所测形状。

容器环境 `pass_with_warnings` 的告警为容器内 `npu-smi` 管理面查询，宿主查询及真实 NPU 计算分别通过。完整精度与长期运行仍需按各模型正式目标验收。
