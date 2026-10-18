# ResWorld：Ascend A3 独立训练与验证


## 离线包快速复验

完整解压本模型最新 ZIP 后，在本目录执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

该入口自动校验完整文件清单、准备固定镜像，并将随包检查点复制到新的输出目录完成独立 NPU 重载与真实数据验证。结果路径由命令输出给出。重新训练及当前验收边界见 [完整验证指南](FULL_VALIDATION.md)。


严格续训、梯度、持续训练、作者权重和原规划/碰撞评测见 [独立扩展验收](FULL_VALIDATION.md)。该指南保留实际失败和正式 full 入口的限制；下文为首次十步功能基线。

## 模型介绍与来源

使用作者 [mengtan00/ResWorld](https://github.com/mengtan00/ResWorld) 的完整网络与 `resworld_config.py`，固定提交 `a5ae973001dd6001d95b3389020a9e263d73a56e`，2026-10-18 核验仍与默认分支 HEAD 一致。网络有 **80,766,238 个参数**，六相机输入、当前帧与两帧历史、R50、256×704 图像、100×100 BEV，输出 3 个导航命令各自的 6 步规划轨迹。

本项目目录参考华为 ModelZoo 的独立 README、test 脚本、训练输出及验证记录组织方式。作者原始 CUDA 配套为 PyTorch 1.9.1、MMCV 1.4.0、MMDetection3D 0.17.1；以下是本项目实际验证的昇腾配套。

## 验证环境与镜像

| 项目 | 本次实测 |
| --- | --- |
| 机器 | 本轮为用户新增 ARM64 Atlas 800I A3；早期1#结果保留历史日期 |
| 卡数 | 本轮4张物理卡，实测逻辑 ID `4,6,12,14`；默认示例 `0,2,4,6` 可按空闲卡调整 |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| 实际 image ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| 官方基镜像 manifest | `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` |
| 配套 | CANN 9.1.0，Python 3.10.21，PyTorch 2.7.1+cpu，TorchNPU 2.7.1.post6，DrivingSDK 26.1.0.post2 |
| MMLab | MMCV 1.7.2，MMDetection 2.24.0，MMDetection3D 1.0.0rc4 |
| 容器资源 | 主机内存上限 96 GiB、16 CPU、8 GiB shm |
| 精度 / 通信 | 显式FP32 / 确定性同步HCCL mean allreduce；按检查点记录核对 |

一张 A3 物理卡有两个逻辑 ID，这里每张物理卡只选择一个。Atlas 800I A3 是推理系列；本次实验成功不改变官方训练硬件支持范围。镜像包含本次验证的头部无显示 OpenCV 依赖，实际镜像归档和 hash 固定于 `images.lock.json`。

## 快速开始

ARM64 Linux 服务器需已安装兼容驱动、Docker 和 Python 3，并有四张空闲物理卡。解压本模型独立 ZIP，进入包含本 README 的目录；完整离线包自带镜像、数据及已验证 checkpoint。GitHub 存放代码和指导，大文件从独立交付包获取。

```bash
cd /var/ai-model/lzd/ResWorld
python3 tools/verify_package.py
bash test/setup.sh
bash test/check_env.sh
bash test/plan.sh
bash test/train_4p.sh
bash test/validate_1p.sh
```

`setup` 校验并加载缺失的镜像，核对实际 image ID，准备本模型内的真实数据与适配代码。由于这套 MMLab 镜像在导入时初始化 NPU，数据准备也会暴露一张卡。`plan` 只输出数据范围及训练参数，不代表 NPU 训练通过。

原交付 checkpoint 可直接复验：

```bash
bash test/validate_1p.sh --output_dir "$PWD/test/output/delivery_verified"
# 四张空闲物理卡的另一个选择
bash test/train_4p.sh --devices 8,10,12,14
# 单卡功能训练
bash test/train_1p.sh --devices 0
```

NPU 容器退出后可能需数秒释放设备。入口记录本模型上一容器退出时间，连续运行自动等待最多 8 秒；Device Resource Busy 时确认本模型上一容器已退出再重试。入口检测占卡会报错；用户业务服务的停机由运行者安排。

## 数据准备与真实监督

离线包的 `data/nuscenes/` 含公开官方 nuScenes mini：323 个 train 帧、81 个 val 帧，8 个训练 scene 与 2 个验证 scene 不重叠。数据还包括同一 mini 的真实 CAN pose / steeranglefeedback（20 个消息文件）、官方 map expansion v1.3、lidarseg-mini 和由作者 depth generator 从真实点云 / lidarseg 投影生成的 2424 个深度监督文件。未以零 CAN、随机图片、随机轨迹或全零深度代替真实输入。

作者 VAD converter 生成训练与验证 infos，`max_sweeps=0`；相机网络使用 keyframe 样本，当前功能包没有全量 sweeps。相机和点云路径改成模型内相对路径，整个目录可移动。每个输入文件的 SHA256、scene 划分及来源记录都在本模型内。

功能训练排除历史不足两帧或未来 6 步不完整的帧；验证使用作者的 `is_train=False` 流程和唯一测试 augmentation，留出 **65 个真实帧**，不使用随机训练增强。用于规划分支的导航命令来自官方标注；没有独立预测该命令。

已有数据执行 `bash test/prepare_data.sh` 会核对全部 hash；缺少监督时在满足 README 目录要求的完整 mini 数据上调用作者 converter / depth generator。数据准备发生摘要不一致时不覆盖旧交付。

## 昇腾适配与训练协议

| 改动 | 实际依据及范围 |
| --- | --- |
| 旧 ROI 接口 import | 当前 MMLab 使用 `mmcv.ops.points_in_boxes_part` 兼容旧名称；完整网络 import / 训练已通过，该方法没有被当前相机规划训练调用，未声明 ROI 算子数值验收 |
| 可变形注意力 | 新训练autograd使用原MMCV公式与原生NPU算子；no_grad推理仍调用DrivingSDK公共API，保持query/heads/sampling配置；CPU使用原MMCV公式 |
| 梯度通信 | 显式启用HCCL确定性，公开DDP hook在NPU同步平均归约；保存配置并记录实际调用，严格续训拒绝跨配置 |
| ResNet MaxPool | 镜像内的 fused MaxPool 无 backward；以可训练 ATen MaxPool 恢复原始 3×3、stride 2、padding 1、ceil_mode=False 几何 |
| 标定矩阵 | 原始 FP64 inverse / matrix product 在 CPU 计算后转 FP32 NPU，避免不受支持的 NPU double 隐式降精度；标定数据无梯度 |
| 数据传送 | 读取原始 DataContainer 后显式传送实际输入 Tensor，避开旧 CUDA scatter stream API；未使用伪造设备状态 |
| CPU Bottleneck 对照 | CPU 的 fused add/relu 恢复为等价 PyTorch 加法和 ReLU，NPU 仍调用原配套实现 |

`vendor/ResWorld` 保留原始固定字节。`tools/prepare_source.py` 在 `runtime/ResWorld` 创建可复核的适配副本，`runtime/patch_manifest.json` 记录改动文件的前后 hash。进程内兼容适配在 `tools/npu_compat.py`，不修改宿主框架安装。

本次功能训练为随机初始化、每 rank batch=1、AdamW lr=1e-4、weight_decay=0.01、gradient clip=35，四卡各 10 个优化器 step（全局累计为 10）。保留作者网络、深度损失和两项规划损失。功能协议未加载 GeoBEV 预训练，也未实现作者完整 12 epoch、500 iteration warmup、EMA 与 SyncBN hook 的验收；full 入口会拒绝运行，避免误称完整训练已可验收。

新 schema2 保存完整游标和各 rank RNG，并核对模型、数据、rank、精度和注意力/通信配置。当前公开入口已完成四卡10＋10与连续20步控制：2054/2054状态张量逐位相同。CPU/NPU梯度仍有未通过项，不能据此宣布整体数值验收通过。使用新输出目录保存初始10步，再从该配置检查点恢复：

```bash
bash test/train_4p.sh --devices 0,2,4,6 --steps 10 \
  --output_dir "$PWD/test/output/first10_new"
bash test/train_4p.sh --devices 0,2,4,6 --steps 20 \
  --resume "$PWD/test/output/first10_new/checkpoint.pt" \
  --output_dir "$PWD/test/output/resumed20_new"
```

## 首次十步功能结果（历史基线）

| 项目 | 本次实测 |
| --- | --- |
| 完整网络前向、反向、参数更新 | 四卡各 10 步；每 rank 446 个有效有限梯度张量 |
| HCCL 参数一致性 | 四个 rank 的全部模型参数 SHA256 相同 |
| checkpoint 保存 / 独立重载 | 权重与优化器有限，strict reload 和保存前参考重放通过 |
| 同权重、同输入 CPU/NPU 对照 | 最大绝对误差 `0.000196456909` |
| 65 帧平均轨迹 L2，1s / 2s / 3s | 1.57227 / 2.66415 / 3.82715 m |
| 3s 终点误差 | 6.76100 m |
| checkpoint SHA256 | `7005808e22a08884bacd10f620b308858f1f0a40aa17dfa2961ff009b2ff0ebf` |

上表为首次十步功能基线，未做碰撞控制。后续已用作者完整权重完成65帧原规划/碰撞CPU/NPU对照，见独立扩展指南；完整trainval、论文精度和生产安全未验收。随机初始化短训练指标不能与作者完整训练精度表比较。

证据：[训练摘要](validation/train_result.json)、[rank 0](validation/rank_0.json)、[rank 1](validation/rank_1.json)、[rank 2](validation/rank_2.json)、[rank 3](validation/rank_3.json)、[独立重载与验证](validation/validation_result.json)、[数据范围](validation/data_summary.json)、[显式 patch](validation/patch_manifest.json)。

另已在服务器新的独立目录完成 setup / checkpoint 严格重载 / CPU 对照 / 65 帧真实留出验证，检查点 SHA 与指标均与原运行相同；证据见 [换目录复验](validation/independent_restore.json)。

## 输出、成功判断与许可

新运行分别保存在 `test/output/<运行名>/`；训练成功才把 `latest_run.json` 标为 completed。已验证交付保存于 `test/output/delivery_verified/`。训练必须有四个 rank 的 1–10 step、有限真实损失和梯度、参数更新、全部参数一致；独立验证 JSON 的 result 应为 `pass_checkpoint_reload_and_validation`。任何门禁失败不作为通过。

本模型独立离线包提供代码、README、data、images、checkpoint、必要验证 JSON、`package_manifest.json` 和 `SHA256SUMS`。原始 ResWorld 源码许可 Apache-2.0 保留于 `vendor/ResWorld/LICENSE`；新增适配遵循本目录 LICENSE。数据遵循各自原始数据使用条款，来源见 `assets.lock.json`。

## 2026-10-18 第二台 A3 机器复验

在相同固定镜像下，重新执行真实数据 4 张物理卡各 10 次优化器更新、原交付检查点严格重载与指标复现、新检查点重载 / 同输入 CPU 对照和真实留出评估，均通过。新训练 CPU/NPU 最大绝对误差 `0.000349164009094`。完整新记录见 [新机复验](validation/revalidation_20261018.json)。本页前面的指标表保留首次迁移基线，新训练指标在该 JSON 单独记录，不保证跨机器逐 bit 复现。

初次四卡启动暴露共享 `runtime/patch_manifest.json.partial` 的并发重命名竞态。现对整个源码 / 清单准备使用模型内 `fcntl.LOCK_EX`，四进程 × 16 轮共 64 次真实准备与适配 SHA 检查通过，修复后完整四卡训练与新检查点验证通过。网络、算子、数据和镜像没有因这个文件锁修复改变。

可在 ARM64 Linux 主机执行不使用 NPU 的并发回归：

```bash
python3 test/test_prepare_source_concurrency.py --workers 4 --rounds 16 --output "$PWD/test/output/source_concurrency.json"
```

本节是第二台机器十步功能复验及文件锁修复的历史记录。随后扩展的严格恢复、梯度、吞吐、30分钟与作者权重评测按独立指南逐项判定；正式trainval/12epoch与驾驶业务未验收，CPU/NPU梯度失败仍需解决。
