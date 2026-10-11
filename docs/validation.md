# 1# 实测结果（2026-10-11）

机器实测为 Atlas 800I A3、8 张双芯片物理卡。两卡 / 四卡试验均选择不同物理卡的一个逻辑芯片，每卡一个 rank；四卡结果来自 1#。这是实验性功能验证，官方训练硬件支持边界见[环境指南](environment.md)。

## 模型结果

| 模型 | 输入和运行范围 | 实际检查结果 | 未覆盖内容 |
| --- | --- | --- | --- |
| BEVFormer | 2 物理卡、公开 nuScenes mini 323 条训练标注，1 epoch / 162 迭代 | 记录的 loss 有限，checkpoint 可读取，1089 模型张量 / 557 优化器状态 | CAN bus 扩展缺失、404 条记录填零；完整 mAP/NDS、性能和稳定性未测 |
| Sparse4D | 2 物理卡、公开 nuScenes mini，10 迭代 | 10 步 loss 有限，checkpoint 可读取，819 模型张量 / 652 优化器状态 | 首末 loss 为 37.1234 / 41.4400，不能称收敛；完整指标未测 |
| Qwen3-VL-8B | 2 物理卡、完整官方权重，6 条演示图文记录，BF16 LoRA 10 步 | 504 个适配器张量有限、252 个 LoRA B 非零、优化器 step=10、两 rank RNG 保存；重新加载后正确识别独立红色正方形图片 | 单张合成图仅作内容检查；业务精度、性能和稳定性未测 |
| WorldEngine | 4 物理卡、官方 NAVTEST 三段日志 / 140 帧 / 64 个时序锚点，真实图像及 PDM 缓存，原图像流水线；官方感知基座，FP32 10 步 | 四 rank 的 10 步原始输出、掩码处理前 loss 和各 259 个梯度张量有限；被检查参数更新及跨 rank 一致；889 模型张量 / 259 优化器状态有限，step=10 | NAVTEST 复用为功能训练，无独立指标；完整 navtrain、PDM 指标、SimEngine、强化学习、性能及稳定性未测 |
| Apollo | 4 物理卡、官方 VAL 与 nuScenes mini 交集 162 帧及对应真实 OCC / CAN bus；原三帧六相机与图像流水线，随机初始化，FP32 10 步 | 四 rank 原始输出、掩码处理前 loss 和各 465 个梯度张量有限；被检查查询参数更新及跨 rank 一致；600 模型张量 / 465 优化器状态有限，step=10 | VAL 复用无独立指标；完整 OCC 归档、官方深度预训练初始化、mAP/mIoU、性能及稳定性待补齐或未测 |

前三项由实机训练和 checkpoint 检查整理。后两项通过公开 `quickstart.sh <模型> mini` 正常退出，保留各 rank 真实训练 JSON：[WorldEngine rank 0](../verification/host1_20261011/worldengine_real/rank0.json)、[rank 1](../verification/host1_20261011/worldengine_real/rank1.json)、[rank 2](../verification/host1_20261011/worldengine_real/rank2.json)、[rank 3](../verification/host1_20261011/worldengine_real/rank3.json)；[Apollo rank 0](../verification/host1_20261011/apollo_real/rank0.json)、[rank 1](../verification/host1_20261011/apollo_real/rank1.json)、[rank 2](../verification/host1_20261011/apollo_real/rank2.json)、[rank 3](../verification/host1_20261011/apollo_real/rank3.json)。对应镜像、源码、补丁与检查代码 SHA256 见 [WorldEngine run](../verification/host1_20261011/worldengine_real/run.json) 和 [Apollo run](../verification/host1_20261011/apollo_real/run.json)。原始日志保留在私有验证目录。

此前的低分辨率合成输入检查作为历史阶段证据保留：[WorldEngine 合成 rank 0](../verification/host1_20261011/worldengine/rank0.json)、[Apollo 合成 rank 0](../verification/host1_20261011/apollo/rank0.json)。它们不代表真实训练；Apollo 的历史合成检查也未覆盖后来修复的编译 FocalLoss 路径。

## 真实数据及适配边界

WorldEngine 使用发布方固定快照的 NAVTEST 标注、PDM 缓存、camera30 图像归档和感知基座 checkpoint，四文件合计 5589709908 字节，均通过发布方 SHA256 校验；完整轨迹词表另为 3932288 字节。数据准备保留连续历史 / 未来帧、原坐标和轨迹标签，仅筛选样本并重写图像路径。实机 [manifest](../verification/host1_20261011/worldengine_real/data_manifest.json)记录 64 个训练锚点、零独立验证样本；官方基座的 backbone / encoder 抽查张量与加载后模型完全相同。范围仅为 AlgEngine imitation learning，不包含完整 WorldEngine 系统。

Apollo 使用官方 VAL PKL 与公开 mini 的 162 帧交集，CAN bus 均非零，稀疏 OCC 文件逐成员检查，复制前后 SHA256 一致。[manifest](../verification/host1_20261011/apollo_real/data_manifest.json)明确 `full_archive_checked=false`：这些完整标签来自官方归档已下载的前段，**15.84 GB 全归档未校验成功**，官方 Google Drive 当前返回下载额度耗尽。完整 OCC 数据等待补齐，不能把此次部分数据说成完整数据训练。PKL SHA256 为本次官方下载观测值，未宣称发布方公布校验值。

Apollo 原编译版 FocalLoss 在原始训练入口发生 `PyThreadState_Get` 致命崩溃。兼容补丁使用 MMDetection 已有的 PyTorch focal-loss 公式，并保留损失参数和 Apollo 类型映射；FP32 CPU/NPU 前后向 8 个案例通过 `atol=1e-5, rtol=1e-4`。最大梯度绝对误差约 `6.20e-6`，sum loss 最大绝对误差 `0.001953125` 满足相对误差标准。[算子对比证据](../verification/host1_20261011/apollo_real/focal_parity.json)仅覆盖这些案例，未替代 CUDA 对照或完整精度回归。

## 保存文件

以下哈希来自实机保存产物；权重和 checkpoint 本体不上传 GitHub。

WorldEngine / Apollo 的真实训练 `iter_10.pth` 包含优化器状态，已在匹配的 NPU 环境读取并检查有限值和 step=10。SDK 的 `meta.iter` 为从 0 计数的 9；文件名的 10 和优化器 step=10 才表示十次更新。未验证从 checkpoint 继续训练。历史 `fixture_step1.pth` 仅含合成检查的模型状态。

| 模型 | 文件 | 字节数 | SHA256 |
| --- | --- | ---: | --- |
| BEVFormer | `outputs/bevformer/epoch_1.pth` | 827201103 | `d50e854d079d28e1c09ef9c57d4959236ca7b9252198da971b24fa174544d70d` |
| Sparse4D | `outputs/sparse4d/iter_10.pth` | 577168640 | `acf68ff01ce0f865532f8ffd55df44490185bdcf2e733d95365402317ea4167d` |
| Qwen | `outputs/qwen3vl8b/adapter_model.safetensors` | 87368144 | `7ee34281193045e042789faa20a570fb6cf3ddc16df1c8aee0923c5640839c7e` |
| WorldEngine | `outputs/worldengine/mini/iter_10.pth` | 434886101 | `4ed2a4ca77ce9252964292faa604154a87a49d21197bf5fd05727caabff83ad8` |
| Apollo | `outputs/apollo_visionnet/mini/iter_10.pth` | 844544012 | `5288adfd7555740da0a41ba04fd2bad30d866a6b9e628fe948d4bf628b576c3d` |

## 镜像和基础检查

DrivingSDK headless 实际 ID 为 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`；Qwen 实际 ID 为 `sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8`；WorldEngine 实际 ID 为 `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3`。完整来源及版本见[环境指南](environment.md)。

DrivingSDK 的双 rank / 四 rank 基础前后向、参数更新和 HCCL all-reduce 均通过，sum 分别为 3 / 10。多尺度变形注意力的前向和 value / location / attention 梯度与参考实现 allclose，最大绝对误差分别为 1.192e-6 / 2.384e-7 / 1.144e-5 / 2.861e-6，仅覆盖测试形状。

容器环境报告为 `pass_with_warnings` 时，当前告警是容器内 `npu-smi` 管理面查询；宿主查询及真实 NPU 计算分别通过。环境导入、基础算子、模型合成检查、真实训练和精度指标应分别判断。
