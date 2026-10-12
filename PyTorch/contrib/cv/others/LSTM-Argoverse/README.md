# LSTM-Argoverse：Ascend A3 独立训练与验证

## 模型介绍

Argoverse 1 车辆二维轨迹预测。使用官方基础 LSTM 编码器 / 解码器，观察 20 帧，递归预测未来 30 帧位置。

来源：[所选上游实现](https://github.com/jagjeet-singh/argoverse-forecasting)，固定提交 `27696a19b3753a4ad8dcae3ad9c6107c5e4e237c`。工程沿用华为 ModelZoo 的独立模型目录、README、`test/` 启动脚本和结果输出方式；本目录是本项目迁移实现。当前验证结果以 `validation/` 内实际记录为准。

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
cd /var/ai-model/lzd/LSTM-Argoverse
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

官方 Argoverse 1.1 完整 train / val 归档中固定选取的 128 / 32 条轨迹。归档、CSV 和生成的 NPZ 均有 SHA256；训练 / 验证 sequence 不重叠。每条 AGENT 保留 50 个真实时刻，坐标归一化与逆变换使用上游函数。历史输入归一化独立核对，不使用未来轨迹确定历史变换。

`bash test/setup.sh` 校验并准备 `data/raw/argoverse_official_functional_subset.tar.gz`。完整上游归档没有打进功能包。子集内的 `source_manifest.json` 记录原归档与每条 CSV 的来源和 hash。

数据及输出路径必须在本模型目录内，便于整个目录迁移。复用已经准备的数据时先核对 hash；出现摘要不一致不要覆盖旧数据。

## 训练与适配

保留原生 `nn.LSTMCell`。当前固定镜像中，NPU batch=1 出现重复执行不一致；适配将一条独立轨迹复制补成 batch=2，再在损失和指标前去掉补齐行。实际前向、14 个参数梯度及独立 checkpoint 重载已与原始 CPU batch=1 对照。CPU/NPU 梯度最大误差为 1.49e-8。训练 batch=8 不触发补齐。上游完整 1/10/30 步 curriculum 未验收，full 模式明确拒绝执行。

默认功能运行：每 rank batch=8，lr=0.001，双卡各执行 10 次优化器更新。10 次更新是整个 DDP 训练的 10 个全局 step，不相加成 20 步。

```bash
# 更换两张空闲物理卡的宿主逻辑 ID
bash test/train_2p.sh --devices 4,6
# 使用单张物理卡做功能训练
bash test/train_1p.sh --devices 0
# 继续训练同一数据 / 同一 rank 数的 checkpoint，steps 为目标累计更新数
bash test/train_2p.sh --resume "$PWD/test/output/delivery_verified/checkpoint.pt" --steps 20
```

续训接口恢复权重和优化器状态；本轮未做续训验收，也不承诺随机数 / sampler 状态逐 bit 重放。保留旧输出，每次新运行会建立新目录；指定 `--output_dir` 时必须选择不存在的本模型内目录。

## 已执行验证

| 检查 | 本次实测结果 |
| --- | --- |
| 真实数据、NPU 前向 / 反向 / 参数更新 | 双卡各 10 步通过 |
| HCCL 参数一致性 | 两个 rank 的全部参数 SHA256 一致 |
| checkpoint | 保存模型及优化器状态，独立进程 strict reload 通过 |
| 重放与 CPU 对照 | 同一 checkpoint / 同一留出输入，最大绝对误差 `5.96046448e-08` |
| 留出集指标 | ADE=13.7612 m，FDE=26.4804 m，Miss Rate=1.0（阈值 2 m，K=1），32 条留出轨迹。 |
| checkpoint SHA256 | `95055ffbbc17978d818ec8ce7167f34583259563a7638bb43e9ad115953d947d` |

证据：[训练摘要](validation/train_result.json)、[rank 0](validation/rank_0.json)、[rank 1](validation/rank_1.json)、[独立验证](validation/validation_result.json)。相同权重、相同输入的 CPU/NPU 数值对照和任务精度是分别检查的两项。

这是随机初始化、10 次优化器更新的功能检查，轨迹质量差，不能用于驾驶决策或作为论文精度复现。尚未完成完整数据训练、curriculum、业务精度、性能及长期稳定性验收。

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

原始网络与坐标处理代码遵循 vendor/LICENSE.Argoverse 中的 BSD 3-Clause Clear；本项目新增适配脚本采用根 LICENSE 的 Apache-2.0。
