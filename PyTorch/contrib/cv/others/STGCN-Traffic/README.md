# STGCN-Traffic：Ascend A3 独立训练与验证


## 离线包快速复验

完整解压本模型最新 ZIP 后，在本目录执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

该入口自动校验完整文件清单、准备固定镜像，并将随包检查点复制到新的输出目录完成独立 NPU 重载与真实数据验证。结果路径由命令输出给出。重新训练及当前验收边界见 [完整验证指南](FULL_VALIDATION.md)。


最新完整训练结果与入口见 [完整训练与验收](FULL_VALIDATION.md)。下文的十步指标为首次功能基线。

## 模型介绍

道路图上的交通速度预测。使用交通 STGCN，输入 12 个 5 分钟历史窗口，预测 15 分钟后的速度。

来源：[所选上游实现](https://github.com/hazdzz/STGCN)，固定提交 `5cfa82c78984fc58fafe4841c35618c6647c4219`。工程沿用华为 ModelZoo 的独立模型目录、README、`test/` 启动脚本和结果输出方式；本目录是本项目迁移实现。当前验证结果以 `validation/` 内实际记录为准。

## 验证环境与镜像

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

## 快速开始

适用条件：ARM64 Linux Ascend 服务器，Docker、宿主驱动及 Python 3 可用，两张物理卡空闲。使用交付的本模型独立 ZIP，解压后进入含本 README 的目录；目录内已包含镜像、数据与检查点。GitHub 仅存放代码和指导，运行所需大文件由独立离线包提供。

```bash
cd /var/ai-model/lzd/STGCN-Traffic
python3 tools/verify_package.py
bash test/setup.sh
bash test/check_env.sh
bash test/plan.sh
bash test/train_2p.sh
bash test/validate_1p.sh
```

`setup` 校验并导入缺失的镜像，然后准备本模型数据；镜像已存在时核对 image ID。`plan` 只核对数据和输出运行参数，不代表训练完成。训练成功后会更新 `test/output/latest_run.json`，验证在独立容器中严格重载最近一次成功训练的 checkpoint。原有交付检查点可直接执行 `bash test/validate_1p.sh --output_dir "$PWD/test/output/delivery_verified"` 复验。

每次 NPU 容器退出后，设备释放可能需要几秒。入口记录本模型上一容器退出时间，连续运行自动等待最多 8 秒；出现 Device Resource Busy，先确认上一次本模型容器已退出并稍后重试。脚本检测卡被占用会报错，工程入口不会擅自停止其他业务服务。

## 数据准备

原论文作者发布的真实 PeMSD7(M)，12672 个时间点、228 个道路传感器。按所选 PyTorch 实现的 70% / 15% / 15% 时间顺序划分，功能模式取 train 前 512 行、val / test 各自前 256 行。归一化参数只由训练行拟合，三段原始时间区间不重叠。功能验证使用 241 个 val 窗口；test 没有用于本次指标。

`bash test/setup.sh` 使用本模型 `data/raw/PeMSD7_Full.zip` 和 `adj.npz`，生成 `data/functional/` 的窗口、训练集 scaler 和图矩阵。原作者 TensorFlow 仓库是数据来源；本模型网络使用注明来源的 PyTorch 社区实现。

数据及输出路径必须在本模型目录内，便于整个目录迁移。复用已经准备的数据时先核对 hash；出现摘要不一致不要覆盖旧数据。

## 训练与适配

将图移位矩阵注册为不持久化 buffer，使 `.to(npu)` 和 CPU 对照使用相同图矩阵；图计算公式与 Chebyshev 图卷积保持所选上游实现。HCCL DDP 启用 find_unused_parameters，保留上游未使用的 Align 参数。

默认功能运行：每 rank batch=8，lr=0.001，双卡各执行 10 次优化器更新。10 次更新是整个 DDP 训练的 10 个全局 step，不相加成 20 步。

```bash
# 更换两张空闲物理卡的宿主逻辑 ID
bash test/train_2p.sh --devices 4,6
# 使用单张物理卡做功能训练
bash test/train_1p.sh --devices 0
# 继续训练同一数据 / 同一 rank 数的 checkpoint，steps 为目标累计更新数
bash test/train_2p.sh --resume "$PWD/test/output/delivery_verified/checkpoint.pt" --steps 20
```

上面的旧功能续训入口恢复权重和优化器状态；完整训练使用 `latest.pt` 恢复 epoch、优化器、调度器、早停和各 rank RNG，已完成独立续训数值控制，详见完整训练指南。保留旧输出，每次新运行会建立新目录；指定 `--output_dir` 时必须选择不存在的本模型内目录。

## 首次十步功能结果（历史基线）

| 检查 | 本次实测结果 |
| --- | --- |
| 真实数据、NPU 前向 / 反向 / 参数更新 | 双卡各 10 步通过 |
| HCCL 参数一致性 | 两个 rank 的全部参数 SHA256 一致 |
| checkpoint | 保存模型及优化器状态，独立进程 strict reload 通过 |
| 重放与 CPU 对照 | 同一 checkpoint / 同一留出输入，最大绝对误差 `7.2658062e-05` |
| 留出集指标 | MAE=7.46664，RMSE=10.90627，MAPE=18.46566%，241 个 val 窗口。MAE/RMSE 使用原始速度数值单位；不擅自标注为 km/h 或 mph。MAPE 排除绝对真值 ≤1e-6 的位置。 |
| checkpoint SHA256 | `420ef3aec85224d9192105e3fd28f3f0775043c888260fec1ed34e6d6b62d914` |

证据：[训练摘要](validation/train_result.json)、[rank 0](validation/rank_0.json)、[rank 1](validation/rank_1.json)、[独立验证](validation/validation_result.json)。相同权重、相同输入的 CPU/NPU 数值对照和任务精度是分别检查的两项。

首次功能子集不能证明全量或论文精度。后续已完成本项目全量 PeMSD7(M) 训练和独立 test，详见完整训练指南；METR-LA、其他 PEMS 数据和原论文 60% / 20% / 20% 协议未因此验收。

另已在服务器新的独立目录完成 setup / checkpoint 严格重载 / CPU 对照 / 真实留出验证，检查点 SHA 与指标均与原运行相同；证据见 [换目录复验](validation/independent_restore.json)。

## 文件与成功判断

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

## 许可

PyTorch 网络遵循 vendor/LICENSE.STGCN 的 LGPL-2.1；原作者来源许可保留于 vendor/LICENSE.STGCN_author。本项目新增脚本采用 Apache-2.0，相关修改源代码随包提供。

## 完整 PeMSD7(M) 训练

已新增完整 epoch、全局 batch=32、验证选优、patience=10 早停、独立 test 和完整续训状态，并完成 52 epoch 实机训练。参数、命令、逐项证据及范围见 [完整训练与验收](FULL_VALIDATION.md)。

## 2026-10-18 第二台 A3 机器复验

在相同固定镜像下，重新执行真实数据 2 张物理卡各 10 次优化器更新、原交付检查点严格重载与指标复现、新检查点重载 / 同输入 CPU 对照和真实留出评估，均通过。新训练 CPU/NPU 最大绝对误差 `8.30665230751e-05`。完整新记录见 [新机复验](validation/revalidation_20261018.json)。本页前面的指标表保留首次迁移基线，新训练指标在该 JSON 单独记录，不保证跨机器逐 bit 复现。

本节是第二台机器的十步功能复验历史。随后新增的完整训练与验收见独立指南；正式业务精度和官方训练硬件支持须按各自证据判断。
