# ResWorld：Ascend A3 独立训练与验证

## 模型介绍与来源

使用作者 [mengtan00/ResWorld](https://github.com/mengtan00/ResWorld) 的完整网络与 `resworld_config.py`，固定提交 `a5ae973001dd6001d95b3389020a9e263d73a56e`，2026-10-13 核验仍为默认分支最新提交。网络有 **80,766,238 个参数**，六相机输入、当前帧与两帧历史、R50、256×704 图像、100×100 BEV，输出 3 个导航命令各自的 6 步规划轨迹。

本项目目录参考华为 ModelZoo 的独立 README、test 脚本、训练输出及验证记录组织方式。作者原始 CUDA 配套为 PyTorch 1.9.1、MMCV 1.4.0、MMDetection3D 0.17.1；以下是本项目实际验证的昇腾配套。

## 验证环境与镜像

| 项目 | 本次实测 |
| --- | --- |
| 机器 | 用户提供的 1#，ARM64 Atlas 800I A3 |
| 卡数 | 4 张物理卡，宿主逻辑 ID `0,2,4,6` |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| 实际 image ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| 官方基镜像 manifest | `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` |
| 配套 | CANN 9.1.0，Python 3.10.21，PyTorch 2.7.1+cpu，TorchNPU 2.7.1.post6，DrivingSDK 26.1.0.post2 |
| MMLab | MMCV 1.7.2，MMDetection 2.24.0，MMDetection3D 1.0.0rc4 |
| 容器资源 | 主机内存上限 96 GiB、16 CPU、8 GiB shm |
| 精度 / 通信 | FP32 / 原生 HCCL DDP |

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
| 可变形注意力 | NPU 明确调用 DrivingSDK 公共 `multi_scale_deformable_attn`，保持原始 query / heads / sampling 配置；CPU 使用 MMCV 原始 PyTorch 对照 |
| ResNet MaxPool | 镜像内的 fused MaxPool 无 backward；以可训练 ATen MaxPool 恢复原始 3×3、stride 2、padding 1、ceil_mode=False 几何 |
| 标定矩阵 | 原始 FP64 inverse / matrix product 在 CPU 计算后转 FP32 NPU，避免不受支持的 NPU double 隐式降精度；标定数据无梯度 |
| 数据传送 | 读取原始 DataContainer 后显式传送实际输入 Tensor，避开旧 CUDA scatter stream API；未使用伪造设备状态 |
| CPU Bottleneck 对照 | CPU 的 fused add/relu 恢复为等价 PyTorch 加法和 ReLU，NPU 仍调用原配套实现 |

`vendor/ResWorld` 保留原始固定字节。`tools/prepare_source.py` 在 `runtime/ResWorld` 创建可复核的适配副本，`runtime/patch_manifest.json` 记录改动文件的前后 hash。进程内兼容适配在 `tools/npu_compat.py`，不修改宿主框架安装。

本次功能训练为随机初始化、每 rank batch=1、AdamW lr=1e-4、weight_decay=0.01、gradient clip=35，四卡各 10 个优化器 step（全局累计为 10）。保留作者网络、深度损失和两项规划损失。功能协议未加载 GeoBEV 预训练，也未实现作者完整 12 epoch、500 iteration warmup、EMA 与 SyncBN hook 的验收；full 入口会拒绝运行，避免误称完整训练已可验收。

续训接口恢复权重 / 优化器并核对模型、数据和 rank 数；本轮未做续训验收，随机数及 sampler 逐 bit 重放也未验收：

```bash
bash test/train_4p.sh --resume "$PWD/test/output/delivery_verified/checkpoint.pt" --steps 20
```

## 实机验证结果

| 项目 | 本次实测 |
| --- | --- |
| 完整网络前向、反向、参数更新 | 四卡各 10 步；每 rank 446 个有效有限梯度张量 |
| HCCL 参数一致性 | 四个 rank 的全部模型参数 SHA256 相同 |
| checkpoint 保存 / 独立重载 | 权重与优化器有限，strict reload 和保存前参考重放通过 |
| 同权重、同输入 CPU/NPU 对照 | 最大绝对误差 `0.000196456909` |
| 65 帧平均轨迹 L2，1s / 2s / 3s | 1.57227 / 2.66415 / 3.82715 m |
| 3s 终点误差 | 6.76100 m |
| checkpoint SHA256 | `7005808e22a08884bacd10f620b308858f1f0a40aa17dfa2961ff009b2ff0ebf` |

规划指标按命令选择一条轨迹、累计预测 offset，在前 2 / 4 / 6 个点计算平均欧氏误差，保留全部有效未来帧。**没有完成碰撞率、论文整套指标、正式任务精度或生产安全验收**。随机初始化 10 步的数值不能与作者完整训练精度表比较。

证据：[训练摘要](validation/train_result.json)、[rank 0](validation/rank_0.json)、[rank 1](validation/rank_1.json)、[rank 2](validation/rank_2.json)、[rank 3](validation/rank_3.json)、[独立重载与验证](validation/validation_result.json)、[数据范围](validation/data_summary.json)、[显式 patch](validation/patch_manifest.json)。

另已在服务器新的独立目录完成 setup / checkpoint 严格重载 / CPU 对照 / 65 帧真实留出验证，检查点 SHA 与指标均与原运行相同；证据见 [换目录复验](validation/independent_restore.json)。

## 输出、成功判断与许可

新运行分别保存在 `test/output/<运行名>/`；训练成功才把 `latest_run.json` 标为 completed。已验证交付保存于 `test/output/delivery_verified/`。训练必须有四个 rank 的 1–10 step、有限真实损失和梯度、参数更新、全部参数一致；独立验证 JSON 的 result 应为 `pass_checkpoint_reload_and_validation`。任何门禁失败不作为通过。

本模型独立离线包提供代码、README、data、images、checkpoint、必要验证 JSON、`package_manifest.json` 和 `SHA256SUMS`。原始 ResWorld 源码许可 Apache-2.0 保留于 `vendor/ResWorld/LICENSE`；新增适配遵循本目录 LICENSE。数据遵循各自原始数据使用条款，来源见 `assets.lock.json`。
