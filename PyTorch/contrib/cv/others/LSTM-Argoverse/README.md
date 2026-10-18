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

## 当前结果

| 检查 | 当前实测结果 |
| --- | --- |
| 完整训练 | 全部 205,942 train / 39,472 val；原 1/10/30 curriculum，725 epoch / 292,175 更新。 |
| 数值与梯度 | 全部 2,368,320 个验证坐标越界 0；最终权重 14 个有效梯度通过。 |
| 严格续训 | 早期 60 张量、成熟状态 102 张量控制均逐位相同。 |
| 未覆盖 | val 同时参与选模；没有独立排行榜 test、驾驶业务或生产性能验收。 |

当前判定：`pass_full_data_curriculum_and_extended_checks`。依据：[逐项实机结果](validation/extended_acceptance_20261018.json)、[完整训练与验证指导](FULL_VALIDATION.md)。以下十步表格保留首次功能基线，不与本次完整或扩展结果混用。

## 首次十步功能结果（历史基线）

| 检查 | 本次实测结果 |
| --- | --- |
| 真实数据、NPU 前向 / 反向 / 参数更新 | 双卡各 10 步通过 |
| HCCL 参数一致性 | 两个 rank 的全部参数 SHA256 一致 |
| checkpoint | 保存模型及优化器状态，独立进程 strict reload 通过 |
| 重放与 CPU 对照 | 同一 checkpoint / 同一留出输入，最大绝对误差 `5.96046448e-08` |
| 留出集指标 | ADE=13.7612 m，FDE=26.4804 m，Miss Rate=1.0（阈值 2 m，K=1），32 条留出轨迹。 |
| checkpoint SHA256 | `95055ffbbc17978d818ec8ce7167f34583259563a7638bb43e9ad115953d947d` |

证据：[训练摘要](validation/train_result.json)、[rank 0](validation/rank_0.json)、[rank 1](validation/rank_1.json)、[独立验证](validation/validation_result.json)。相同权重、相同输入的 CPU/NPU 数值对照和任务精度是分别检查的两项。

上表是首次随机初始化、10 次更新的历史功能检查，轨迹质量差。后续完整数据训练、curriculum、全量数值、恢复、吞吐和长时间运行已另行实测，见完整训练指南；业务精度和独立排行榜未验收。

另已在服务器新的独立目录完成 setup / checkpoint 严格重载 / CPU 对照 / 真实留出验证，检查点 SHA 与指标均与原运行相同；证据见 [换目录复验](validation/independent_restore.json)。

## 2026-10-18 新机功能复验（历史）

在相同固定镜像下，重新执行真实数据 2 张物理卡各 10 次优化器更新、原交付检查点严格重载与指标复现、新检查点重载 / 同输入 CPU 对照和真实留出评估，均通过。新训练 CPU/NPU 最大绝对误差 `5.96046447754e-08`。完整新记录见 [新机复验](validation/revalidation_20261018.json)。本页前面的指标表保留首次迁移基线，新训练指标在该 JSON 单独记录，不保证跨机器逐 bit 复现。

补充真实 batch=1 的 CPU / NPU 补齐路径前向、loss 和 14 参数梯度对照通过，最大梯度误差 `1.49011611938e-08`，见 [新机 B1 控制](validation/batch1_control_20261018.json)。未重定位原生 B1 底层机制。

本节为第二台机器的十步功能复验历史；后续完整训练及验证使用独立 full 入口和结果，见完整训练指南。正式业务精度与官方训练硬件支持不能由功能结果推定。

## 输出文件与通过判据

| 位置 | 用途 |
| --- | --- |
| `test/` | 本模型环境检查、训练和独立验证脚本 |
| `tools/`、`vendor/` | 本模型适配及固定源码 |
| `data/` | 本模型原始和预处理数据，离线包中提供 |
| `images/` | 本模型运行镜像，离线包中提供 |
| `test/output/delivery_verified/` | 本次已验证 checkpoint、重载输入和摘要，离线包中提供 |
| `validation/` | 可公开的验证结果 |
| `package_manifest.json`、`SHA256SUMS` | 独立离线包的文件完整性清单 |

训练日志应有每个 rank 的 1–10 step，`train_result.json` 为 `pass_functional_training`；独立验证产生 `validation_result.json`，其结果为 `pass_checkpoint_reload_and_validation`。退出码非 0、非有限 loss / 梯度、参数未更新、rank hash 不一致、重载或 CPU 对照失败，均不得判定通过。

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
