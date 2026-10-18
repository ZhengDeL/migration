# LSTM-Argoverse for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

Argoverse 1 车辆二维轨迹预测。使用官方基础 LSTM 编码器 / 解码器，观察 20 帧，递归预测未来 30 帧位置。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/jagjeet-singh/argoverse-forecasting
  commit_id=27696a19b3753a4ad8dcae3ad9c6107c5e4e237c
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/jagjeet-singh/argoverse-forecasting/blob/27696a19b3753a4ad8dcae3ad9c6107c5e4e237c/README.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

## 迁移说明

保留原生 `nn.LSTMCell`。当前固定镜像中，NPU batch=1 出现重复执行不一致；适配将一条独立轨迹复制补成 batch=2，再在损失和指标前去掉补齐行。实际前向、14 个参数梯度及独立 checkpoint 重载已与原始 CPU batch=1 对照。CPU/NPU 梯度最大误差为 1.49e-8。训练 batch=8 不触发补齐。独立 full 入口已完成原始 1/10/30 步 curriculum：725 epoch、292175 更新；完整数据、成熟状态恢复和最终权重数值/梯度结果见完整训练指南。

默认功能运行：每 rank batch=8，lr=0.001，双卡各执行 10 次优化器更新。10 次更新是整个 DDP 训练的 10 个全局 step，不相加成 20 步。

# 准备训练环境

## 安装昇腾环境

ARM64 Linux、Docker、兼容宿主驱动与两张空闲物理卡；先核对宿主 `npu-smi info`。

## 安装模型环境

实际验证机器为用户提供的 1# ARM64 Atlas 800I A3，使用 2 张物理卡；默认宿主逻辑 ID 为 `0,2`。A3 一张物理卡对应两个逻辑 ID，本项目每张物理卡只选一个。Atlas 800I 属于推理系列，这次实验不改变官方训练硬件支持范围。

| 项目 | 本次实测 |
| --- | --- |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| 镜像 ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| 官方基镜像 manifest | `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` |
| 配套 | CANN 9.1.0 / Python 3.10.21 / PyTorch 2.7.1+cpu / TorchNPU 2.7.1.post6 / DrivingSDK 26.1.0.post2 |
| 容器资源 | 主机内存上限 48 GiB，8 CPU，8 GiB shm |
| 精度与通信 | FP32 / HCCL DDP |

镜像继承官方 DrivingSDK，额外使用无显示依赖的 OpenCV 配套。`images.lock.json` 固定实际 image ID 和离线归档 hash。工程不会因上游出现新 tag 自动更换已经验证的镜像。

## 准备数据集

官方 Argoverse 1.1 完整 train / val 归档中固定选取的 128 / 32 条轨迹。归档、CSV 和生成的 NPZ 均有 SHA256；训练 / 验证 sequence 不重叠。每条 AGENT 保留 50 个真实时刻，坐标归一化与逆变换使用上游函数。历史输入归一化独立核对，不使用未来轨迹确定历史变换。

`bash test/setup.sh` 校验并准备 `data/raw/argoverse_official_functional_subset.tar.gz`。子集内的 `source_manifest.json` 记录原归档与每条 CSV 的来源和 hash。最新独立包另含完整官方 train/val 原始归档和 `data/full/` 缓存；按完整训练指南使用，功能数据与完整数据分别核对摘要。

数据及输出路径必须在本模型目录内，便于整个目录迁移。复用已经准备的数据时先核对 hash；出现摘要不一致不要覆盖旧数据。

## 准备预训练或交付权重

本模型默认从头训练，不依赖外部预训练网络。重载复验使用随包已验证检查点；完整训练的 `checkpoint.pt` 用于推理，`latest.pt` 用于恢复完整训练状态，见 [完整训练与验收](FULL_VALIDATION.md)。

# 开始训练

## 训练模型

适用条件：ARM64 Linux Ascend 服务器，Docker、宿主驱动及 Python 3 可用，两张物理卡空闲。使用交付的本模型独立 ZIP，解压后进入含本 README 的目录；目录内已包含镜像、数据与检查点。GitHub 仅存放代码和指导，运行所需大文件由独立离线包提供。



以下为十步功能入口。进入实际解压目录后顺序执行：

```bash
python3 tools/verify_package.py
bash test/setup.sh
bash test/check_env.sh
bash test/plan.sh
bash test/train_2p.sh
bash test/validate_1p.sh
```



`setup` 校验并导入缺失的镜像，然后准备本模型数据；镜像已存在时核对 image ID。`plan` 只核对数据和输出运行参数，不代表训练完成。训练成功后会更新 `test/output/latest_run.json`，验证在独立容器中严格重载最近一次成功训练的 checkpoint。原有交付检查点可直接执行 `bash test/validate_1p.sh --output_dir "$PWD/test/output/delivery_verified"` 复验。

每次 NPU 容器退出后，设备释放可能需要几秒。入口记录本模型上一容器退出时间，连续运行自动等待最多 8 秒；出现 Device Resource Busy，先确认上一次本模型容器已退出并稍后重试。脚本检测卡被占用会报错，工程入口不会擅自停止其他业务服务。

## 单卡、换卡与历史功能续训

```bash
# 更换两张空闲物理卡的宿主逻辑 ID
bash test/train_2p.sh --devices 4,6
# 使用单张物理卡做功能训练
bash test/train_1p.sh --devices 0
# 继续训练同一数据 / 同一 rank 数的 checkpoint，steps 为目标累计更新数
bash test/train_2p.sh --resume "$PWD/test/output/delivery_verified/checkpoint.pt" --steps 20
```



上述为旧功能检查点的权重/优化器续训。完整训练从 `latest.pt` 恢复两个 Adam、curriculum、各阶段最佳模型和全部 rank RNG；早期 60 张量及成熟状态 102 张量控制均逐位相同。完整选定的 `checkpoint.pt` 用于推理，不能代替 `latest.pt` 续训。每次新运行使用不存在的输出目录。

## 完整数据训练

完整训练按本模型所选原始协议执行。使用新输出目录，`full` 入口不能用十步检查代替：

```bash
bash test/prepare_data.sh --mode full
bash test/train_full_2p.sh --devices 0,2
bash test/validate_1p.sh --devices 0
```

数据数量、优化器、调度／curriculum、早停、续训与成功条件见 [完整训练与验收](FULL_VALIDATION.md)。

## 独立验证

完整解压本模型最新 ZIP 后，在本目录执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

该入口自动校验完整文件清单、准备固定镜像，并将随包检查点复制到新的输出目录完成独立 NPU 重载与真实数据验证。结果路径由命令输出给出。重新训练及当前验收边界见 [完整验证指南](FULL_VALIDATION.md)。


完整数据、curriculum、续训与验收方法见 [完整训练与验收](FULL_VALIDATION.md)。下文保留首次十步功能基线。

# 训练结果展示

## 本项目实测（2026-10-18）

设备为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），驱动 26.1.1 / 固件 9.0.0.9.220；固定镜像及软件版本见本页环境表。

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

当前判定：`pass_full_data_curriculum_and_extended_checks`。

当前主要权重 SHA256：`306ca75daba7116ed6e4cc2dc93df5c20b3ea08d6fe04995072fc957146ff29a`。用途：按 val 选择的最佳模型，不能用于完整状态续训；续训使用 latest 检查点。

## 训练性能实测

| 进程数 | 全局吞吐（样本/s） | 每 rank batch | 预热更新 | 计时更新 |
| --- | --- | --- | --- | --- |
| 1p | 177.918 | 8 | 10 | 50 |
| 2p | 364.430 | 8 | 10 | 50 |

实测包含真实数据加载、传输、前向、反向及优化器更新，各更新同步；使用不同物理卡的单个逻辑芯片。主机 CPU/存储有其他任务共享，没有提供生产性能目标；本表不作为官方 FPS 或性能达标证明。

## 验证依据与复现

[逐项实机结果及原门槛](validation/extended_acceptance_20261018.json)记录实际执行阶段、退出码、输入范围、权重身份、数值/恢复/持续训练及未完成项。复现命令和结果文件见[完整训练与验证指导](FULL_VALIDATION.md)。旧十步基线保留在 `validation/train_result.json` 和 `validation/validation_result.json`，不能覆盖本节当前结果。

恢复测试比较完整模型、优化器及适用的调度器、训练游标和各 rank RNG；单纯能加载模型不等于续训通过。CPU/NPU 使用相同输入与相同权重的数值对照，与数据集任务指标分别报告。

# 版本说明

## 变更

- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示／现象 | 处理 |
| --- | --- |
| 只有 GitHub 源码，缺少镜像或数据 | 取得本模型完整 ZIP，按本页固定镜像和数据清单核对 |
| image ID 或数据摘要不符 | 对照本模型 images.lock.json 和来源锁；使用匹配版本 |
| Device Resource Busy | 核对物理拓扑与占用，等待本模型容器释放后重试 |
| 输出目录已存在 | 为新运行指定本模型内不存在的目录，保留旧结果 |
| 需要精确续训 | 按完整指南选择正确检查点，保持数据、rank、batch 和数值配置一致 |
| 仅有日志没有通过结果 | 核对退出码及本次 JSON，不能将旧结果当本次通过 |

## 许可

原始网络与坐标处理代码遵循 vendor/LICENSE.Argoverse 中的 BSD 3-Clause Clear；本项目新增适配脚本采用根 LICENSE 的 Apache-2.0。
