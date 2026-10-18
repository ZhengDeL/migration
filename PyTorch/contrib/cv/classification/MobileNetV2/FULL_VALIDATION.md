# MobileNetV2 数值、持续训练和 ImageNet 完整验证

2026-10-18 在第二台 A3 机器完成：双卡功能训练、严格续训、梯度/优化器控制、30 分钟训练后独立重载，以及官方 V1 权重的 ImageNet 全 50,000 张 CPU/NPU 评估。600 epoch 从头训练尚未完成。逐项实测见 [本轮验证记录](validation/extended_acceptance_20261018.json)。

## 1. 使用同一模型与明确的权重

网络来自 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) 的 MobileNetV2，保留 width=1.0 和 1,000 类输出。功能训练采用真实 Imagenette 子集映射到原始 ImageNet 类别；完整精度验证使用 ModelZoo `url.ini` 指定的 `mobilenet_v2-b0353104.pth`（Torchvision V1）。两者是不同训练状态，结果分别记录，不能互相代替。

镜像、image ID 和运行方式见 [README](README.md)。本模型目录内提供自己的脚本、源码锁、镜像锁、数据和验证结果，不引用其他模型目录。

## 2. FP32 配置与恢复

`tools/npu_precision.py` 在计算前显式设置 `conv.allow_hf32=False`、`matmul.allow_hf32=False`，并核对实际值。配置进入 checkpoint 的 `settings.npu_precision`；精确续训要求计算精度相同。旧检查点没有该字段时，独立验证保留其原运行默认精度重放，不悄悄切换设置来覆盖旧结果。

2026-10-18 扩展验证发现：默认卷积 HF32 开启时，30 分钟训练后的一批 CPU/NPU 对照有 1/4,000 个 logit 越界。同权同输入关闭 HF32 后，最大差从约 7.05e-3 降至 2.10e-5，越界为 0。原 rtol=atol=0.003 没有放宽。新配置已在新目录重新执行训练、保存参考、续训、梯度、持续训练和独立重载；原始失败保留。

## 3. 功能训练和独立数值验证

```bash
bash test/setup.sh
bash test/train_2p.sh --devices 0,2
bash test/validate_1p.sh --devices 0
python3 tools/validate_extended.py --check gradient --nproc 1 --devices 0 \
  --checkpoint "$PWD/test/output/delivery_verified/checkpoint.pt" \
  --output_dir "$PWD/test/output/gradient_control"
python3 tools/validate_extended.py --check stability --nproc 2 --devices 0,2 \
  --checkpoint "$PWD/test/output/delivery_verified/checkpoint.pt" \
  --duration_seconds 1800 --output_dir "$PWD/test/output/stability_30min"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/stability_30min"
```

schema2 检查点记录下一 epoch/batch 游标、Python/NumPy/Torch/NPU 的全 rank RNG。恢复时在加载和图像增强前跳过已消费批次，逐步记录真实增强输入摘要；使用“10 步保存＋续至 20 步”和“连续 20 步”独立比较，不能以状态字段存在代替恢复正确。

检查点来源按项区分：新 FP32 十步/二十步运行用于严格恢复和独立重载；原固定 `delivery_verified/checkpoint.pt` 在显式 FP32 配置下用于可比较的梯度、吞吐和 30 分钟训练，其源 SHA 为 `b7ae56ada3c51fd9d4c00c9eef1711497e996569d64eb3e27118c77327efbf89`。后者是从该固定模型/优化器状态开始的新精度运行，不声称重现旧精度下的训练轨迹。每项 JSON 记录自己的源 SHA，不能把它们写成同一个检查点。

梯度控制使用同权同批真实图像，在 eval 模式保留 autograd，检查所有有效梯度及同配置优化器的一步更新；真实 train 模式另外做 1,800 秒持续训练。通过要求包括有限 loss/梯度/模型及优化器状态、参数确实更新、各 rank 参数 SHA 一致和保存后的独立重载。吞吐使用 `--check benchmark --nproc 1/2`，记录加载到优化器及同步的完整真实训练开销；没有用户给出的生产性能目标。

### 本轮实测

| 检查 | FP32 配置结果 |
| --- | --- |
| 10＋10 vs 连续 20 步 | 模型/优化器/游标/全 rank RNG 共 478 个张量全部逐位相同，最大差 0；真实增强输入 hash 一致 |
| 固定源权重 CPU/NPU 梯度 | 158 个有效梯度通过；相对 L2=0.0001165356；一步参数最大差 1.1362e-7 |
| 新 FP32 二十步权重梯度 | 158 个有效梯度全部通过；相对 L2=0.00040335445；一步参数最大差 7.2923e-7；源 SHA 为下述 schema2 二十步权重 |
| 1p / 2p 真实训练吞吐 | 87.506 / 153.102 图像每秒，每 rank batch=4；共享机器实测，非生产性能承诺 |
| 双卡持续训练 | 1,800.501374 秒，33,570 次更新；模型/优化器有限、参数更新、各 rank 参数 SHA 一致 |
| 长训练后的独立重载 | 保存参考重放通过，CPU/NPU 最大差 2.4795532e-5 |
| 功能子集质量 | 仅 100 train / 40 val；长训练状态 Top-1=19/40、Top-5=37/40，不代替完整 ImageNet 结果 |

检查点用途不同：`acceptance_resumed20_fp32_20261018/checkpoint.pt` 保存 schema2 的完整游标和全 rank RNG；`acceptance_stability_fp32_20261018/checkpoint.pt` 保存长训练模型与优化器，不包含精确恢复所需的全部游标/RNG，不能用于宣称严格续训相等。上述目录均位于 `test/output/`。独立离线包可先复验后者：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

## 4. 官方 ImageNet 50,000 张完整验证

`imagenet_sources.lock.json` 固定官方归档地址、Torchvision 公布的 MD5、原始 V1 权重 SHA256、预处理和事先确定的项目数值门槛。准备以下三个文件至本模型 `data/imagenet_official/`：

- `ILSVRC2012_devkit_t12.tar.gz`，MD5 `fa75699e90414af021442c21a62c3abf`。
- `ILSVRC2012_img_val.tar`，6,744,924,160 字节，MD5 `29b22e2961454d5413ddabcf34fc5622`。
- `mobilenet_v2-b0353104.pth`，SHA256 `b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`。

```bash
bash test/validate_imagenet_1p.sh --devices 0 \
  --output_dir "$PWD/test/output/imagenet_v1_full_validation"
```

该命令在固定镜像内先完成 CPU 全集参考，再执行 NPU 全集评测；NPU 阶段需要一张空闲物理卡。验证直接读取原始 tar 内 JPEG，避免重复解出 50,000 个文件。标签必须经官方 devkit 数字 ID → WNID → 原 1,000 类索引映射，不能使用 devkit ID−1。映射已与固定镜像中 Torchvision 的独立 devkit 解析器核对，全部 50,000 条一致。

采用 V1 的 Resize(256, BILINEAR)、CenterCrop(224)、原 mean/std；没有换成 V2 权重或其 Resize(232) 配置。完整读取每张图像一次，统计 Top-1/Top-5，并比较 50,000,000 个 CPU/NPU logits。项目门槛：全部 logits 满足 rtol=atol=0.003，Top-1/Top-5 与完整 CPU 参考差不超过 0.1 个百分点。CPU 还与已公布 V1 71.878%/90.286% 对照。上述 0.1 个百分点是本项目工程门槛，不是华为公布的验收标准。

结果保存在 `cpu_result.json`、`npu_result.json` 和进度 JSON。`cpu_logits.npy` 是中间数值参考，不能用文件存在或进度完成代替结果 `pass_official_full_validation`。

本轮两个后端均完整评估 50,000 张图：Top-1=35,935/50,000（71.87%），Top-5=45,147/50,000（90.294%）。CPU/NPU 的 Top-1、Top-5 完全一致；50,000,000 个 logits 全部通过原数值门槛，最大绝对差 6.8664551e-5，越界 0。CPU 相对公布 V1 的 Top-1 / Top-5 各差 0.008 个百分点，通过预设 0.1 个百分点门槛。CPU / NPU 评估用时分别为 1,137.173 / 76.222 秒，受本次环境和并发任务影响，不能当作隔离性能基准。

## 5. 完整训练仍需单独验收

官方预训练权重的 50,000 张完整精度评测不能证明从头训练 600 epoch 的收敛。完整训练还需要全部 1,281,167 张 ImageNet train、50,000 张 val、对应数据布局、官方训练超参数和完整训练运行。`ILSVRC2012_img_train.tar` 为 147,897,477,120 字节，MD5 `1d675b47d978889d74fa0da5fadfb00e`；数据与实测结果未齐前不能宣布该项完成。

Imagenette 100/40 图像功能子集不能通过 full 数据守卫，不能重命名其 metadata 来冒充 ImageNet。正式业务精度、生产吞吐和 Atlas 800I A3 的官方训练支持也要分别确认。

完整逐参数结果见 `validation/gradient_parameters_20261018.json` 与 `validation/selected20_gradient_parameters_20261018.json`；两种源权重分别保存，不把一次通过当成另一份权重的实测。
