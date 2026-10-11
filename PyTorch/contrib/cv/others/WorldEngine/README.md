# WorldEngine AlgEngine：Ascend 训练迁移

## 范围与实测状态

采用 [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine/tree/3eded7e8ac64e5bed817860765089b5914ab698f) 固定提交 `3eded7e8ac64e5bed817860765089b5914ab698f`，覆盖 AlgEngine 的 `e2e_vadv2_50pct.py`：NAVFormer / TrajScoringHead、NPU 分布式入口和多尺度变形注意力。SimEngine、3DGS、强化学习及完整后训练链路尚未适配。

2026-10-11 在 1# 四张不同物理卡上，通过公开 quickstart 入口完成真实数据 10 步训练。使用官方 NAVTEST 三段日志的 140 帧 / 64 个时序锚点、真实图像与 PDM 缓存；原轨迹标签未改写，官方感知基座权重已加载。NAVTEST 复用为功能训练，**不评估独立验证精度**。配置名称的“50%”是上游正式数据划分，当前子集不等于完成其 50% 训练。

## 镜像

实际使用 `local/migration-worldengine:26.1.0-a3-20261011`，image ID `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3`。基于 DrivingSDK headless 镜像，保留 Python 3.10.21、CANN 9.1.0、Torch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。

AlgEngine / nuPlan 依赖隔离安装；Motional 官方 nuPlan 使用固定 Git 提交，NumPy 为 1.23.5，ARM64 地理包使用独立 conda prefix。Dockerfile、固定依赖、镜像来源和新机器构建步骤见[环境指南](../../../../../docs/environment.md)。

## 快速开始

在已准备好镜像和源码的 1#，从仓库根目录执行：

```bash
bash tools/quickstart.sh worldengine check
bash tools/quickstart.sh worldengine download
bash tools/quickstart.sh worldengine plan
bash tools/quickstart.sh worldengine mini
```

`download` 获取约 5.59 GB 官方 NAVTEST 标注、PDM 缓存、感知基座 checkpoint 和 camera30 图像归档，另取约 4 MB 官方轨迹词表；固定发布方提交、大小和 SHA256，自动准备功能子集。已有完整文件校验后复用，分段中断后重跑续传。建议至少有 20 GB 可用空间。已有文件仅需重新准备时用 `worldengine prepare`；1# 已准备完成，可直接从 `plan` 开始。

默认宿主逻辑设备 `0,2,4,6` 对应四张不同物理卡，容器内为 `0,1,2,3`。运行前检查空闲；可用 `ASCEND_HOST_DEVICES` 换卡。`mini` 生成原配置的 FP32 四卡、每卡 batch 1、AdamW 学习率 `5e-5`、10 步功能配置，保留四帧 / 八相机、原图加载与 downsample / resize 流水线、200×200 BEV、六层 encoder 和完整 8192×40×3 轨迹词表。

## 怎么判断通过

`mini` 正常退出，`outputs/worldengine/mini/training_check_rank{0,1,2,3}.json` 都是 `pass_training_smoke`。每个 rank 的 10 步原始输出、掩码处理前 imitation loss 和各 259 个梯度张量有限；参数更新，更新后的被检查规划参数跨 rank 一致。

rank 0 的 `checkpoint_reload` 为 `pass`、`optimizer_step_values` 为 `[10.0]`；`iter_10.pth` 为 434886101 字节，SHA256 `4ed2a4ca77ce9252964292faa604154a87a49d21197bf5fd05727caabff83ad8`，889 个模型张量和 259 个优化器状态均有限。保存检查未包含 checkpoint 续训。报告另确认加载的 `img_backbone.conv1.weight` 和 encoder sampling-offset 权重与官方初始化文件一致。公开逐 rank 证据、来源和检查范围见[实测记录](../../../../../docs/validation.md)。

## 数据和适配

官方数据固定在 [WorldEngine 数据快照](https://huggingface.co/datasets/OpenDriveLab/WorldEngine/tree/d0e88d858533ca407a3387a26577c083aa7b3617) 和 [OpenScene 数据快照](https://huggingface.co/datasets/OpenDriveLab/OpenScene/tree/a76f840b65e972bc45e56c2adced897498e9a026)。四个文件的发布方 SHA256 / 大小见[下载清单代码](../../../../../tools/world_mini_data.py)，实机子集记录见[数据 manifest](../../../../../verification/host1_20261011/worldengine_real/data_manifest.json)。默认数据目录 `/var/ai-model/lzd/data/worldengine_smoke`。

数据准备仅选取原始帧、重写图像绝对路径，并保留连续历史 / 未来帧、坐标、标签和真实 PDM 数据；不启用会改变数据语义的 `diffusiondrive_data_mode`。`ascend.patch` 改训练入口 / 分布式包装 / 变形注意力，`compat.patch` 修正 Matplotlib 风格名。固定补丁 SHA256 见[来源锁定](../../../../../sources.lock.json)。

## 正式训练待办

按固定上游[数据指南](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/data_organization.md)准备完整 OpenScene/NAVSIM 图像、navtrain/navtest 标注、两套 PDM 缓存和基座 checkpoint，然后运行 `bash tools/quickstart.sh worldengine train`。该入口检查正式数据路径；四卡每卡 batch 1、学习率 `5e-5`、8 epoch 为实验起点，需重新评估 batch / 学习率。

`build` 仅构建模型，`compute` 保留 96×160 合成输入检查。真实短训练未覆盖完整 PDM / 开环规划指标、长期稳定性、性能、SimEngine 闭环和 RL，也不改变[官方训练硬件支持](../../../../../docs/environment.md)边界。
