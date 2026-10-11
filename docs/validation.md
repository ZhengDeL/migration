# 1# 实测结果（2026-10-11）

机器实测为 Atlas 800I A3、8 张双芯片物理卡。两卡 / 四卡试验均选择不同物理卡的一个逻辑芯片，每卡一个 rank；四卡结果来自 1#。这是实验性功能验证，官方训练硬件支持边界见[环境指南](environment.md)。

## 模型结果

| 模型 | 输入和运行范围 | 实际检查结果 | 未覆盖内容 |
| --- | --- | --- | --- |
| BEVFormer | 2 物理卡、公开 nuScenes mini 323 条训练标注，1 epoch / 162 迭代 | 记录的 loss 有限，checkpoint 可读取，1089 模型张量 / 557 优化器状态 | CAN bus 扩展缺失、404 条记录填零；完整 mAP/NDS、性能和稳定性未测 |
| Sparse4D | 2 物理卡、公开 nuScenes mini，10 迭代 | 10 步 loss 有限，checkpoint 可读取，819 模型张量 / 652 优化器状态 | 首末 loss 为 37.1234 / 41.4400，不能称收敛；完整指标未测 |
| Qwen3-VL-8B | 2 物理卡、完整官方权重，6 条演示图文记录，BF16 LoRA 10 步 | 504 个适配器张量有限、252 个 LoRA B 非零、优化器 step=10、两 rank RNG 保存；重新加载后正确识别独立红色正方形图片 | 单张合成图仅作内容检查；业务精度、性能和稳定性未测 |
| WorldEngine | 4 物理卡、随机初始化、合成 4 帧 × 8 相机 96×160；200×200 BEV、8192 条官方轨迹词表 | 四 rank 原始输出、掩码处理前 loss 和各 259 个梯度张量有限；参数更新 / 跨 rank 被检查参数一致；checkpoint 保存重载通过 | 真实数据、预训练基座、原始分辨率、PDM 指标、SimEngine、强化学习未测 |
| Apollo | 4 物理卡、随机初始化、合成 3 帧 × 6 相机 96×160；一个检测框及稀疏 OCC 标签 | 四 rank 原始检测 / OCC 输出和各 465 个梯度张量有限；检测查询参数更新；checkpoint 保存重载通过 | 真实 OCC 数据、官方深度预训练权重、正式分辨率、mAP/mIoU 未测 |

前三项由实机训练和 checkpoint 检查整理。后两项保留各 rank 公开 JSON：[WorldEngine rank 0](../verification/host1_20261011/worldengine/rank0.json)、[rank 1](../verification/host1_20261011/worldengine/rank1.json)、[rank 2](../verification/host1_20261011/worldengine/rank2.json)、[rank 3](../verification/host1_20261011/worldengine/rank3.json)；[Apollo rank 0](../verification/host1_20261011/apollo/rank0.json)、[rank 1](../verification/host1_20261011/apollo/rank1.json)、[rank 2](../verification/host1_20261011/apollo/rank2.json)、[rank 3](../verification/host1_20261011/apollo/rank3.json)。原始日志保留在私有验证目录。

## 保存文件

以下哈希来自实机保存产物；权重和 checkpoint 本体不上传 GitHub。

WorldEngine / Apollo 的 `fixture_step1.pth` 仅用于合成计算的模型保存重载检查，不含优化器续训验收。

| 模型 | 文件 | 字节数 | SHA256 |
| --- | --- | ---: | --- |
| BEVFormer | `outputs/bevformer/epoch_1.pth` | 827201103 | `d50e854d079d28e1c09ef9c57d4959236ca7b9252198da971b24fa174544d70d` |
| Sparse4D | `outputs/sparse4d/iter_10.pth` | 577168640 | `acf68ff01ce0f865532f8ffd55df44490185bdcf2e733d95365402317ea4167d` |
| Qwen | `outputs/qwen3vl8b/adapter_model.safetensors` | 87368144 | `7ee34281193045e042789faa20a570fb6cf3ddc16df1c8aee0923c5640839c7e` |
| WorldEngine | `outputs/worldengine/compute_fixture/fixture_step1.pth` | 233338823 | `3b97ae39e728cde5fc0518a714a0d3a0b70fa9e04abd3702c55c1407be6b7288` |
| Apollo | `outputs/apollo_visionnet/compute_fixture/fixture_step1.pth` | 282956419 | `18d2d3f0ab5d5913139ceee2865959b91b7e7b6937043139daea726ac5dc730c` |

## 镜像和基础检查

DrivingSDK headless 实际 ID 为 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`；Qwen 实际 ID 为 `sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8`；WorldEngine 实际 ID 为 `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3`。完整来源及版本见[环境指南](environment.md)。

DrivingSDK 的双 rank / 四 rank 基础前后向、参数更新和 HCCL all-reduce 均通过，sum 分别为 3 / 10。多尺度变形注意力的前向和 value / location / attention 梯度与参考实现 allclose，最大绝对误差分别为 1.192e-6 / 2.384e-7 / 1.144e-5 / 2.861e-6，仅覆盖测试形状。

容器环境报告为 `pass_with_warnings` 时，当前告警是容器内 `npu-smi` 管理面查询；宿主查询及真实 NPU 计算分别通过。环境导入、基础算子、模型合成检查、真实训练和精度指标应分别判断。
