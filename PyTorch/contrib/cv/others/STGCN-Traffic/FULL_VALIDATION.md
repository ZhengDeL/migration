# STGCN-Traffic 完整训练与验收

本指南只对应道路交通速度预测 STGCN。网络采用 [hazdzz/STGCN](https://github.com/hazdzz/STGCN) 的 PyTorch 实现，数据为原作者发布的 PeMSD7(M)。本项目按该实现的 70/15/15 时间划分验证；原论文的 60/20/20 协议和其他交通数据集需分别验证。

2026-10-18 当前显式 FP32 配置已经完成完整训练、严格恢复、全部 val/test 数值、梯度与优化器、双卡 30 分钟及独立重载。逐项实测见 [本轮完整验证记录](validation/extended_acceptance_20261018.json)。

## 1. 环境

使用本模型 [README](README.md) 中固定的 ARM64 镜像，CANN 9.1.0、Torch 2.7.1、TorchNPU 2.7.1.post6；完整 image ID 由 `images.lock.json` 校验。每张 A3 物理卡只取一个逻辑 ID，例如双卡 `0,2`。实测 Atlas 800I A3 推理系列不因此获得官方训练支持。

## 2. 准备并运行

在本模型独立目录执行。完整离线包中的原始数据、镜像和已验证检查点均属于本模型，无需引用其他模型目录。

```bash
bash test/setup.sh
bash test/prepare_data.sh --mode full
bash test/train_full_2p.sh --devices 0,2
bash test/validate_1p.sh --devices 0
```

训练默认全局 batch=32，双卡每 rank=16。每个 epoch 完整覆盖全部 8,857 个训练窗口，最后不完整 batch 按实际全局样本数加权，避免重复或丢弃训练窗口。验证和测试各有 1,885 个窗口。scaler 只拟合训练时间段，跨划分不构造相交窗口。

训练采用原 Chebyshev 图卷积、dropout=0.5、AdamW lr=0.001/weight decay=0.001、StepLR(step_size=10, gamma=0.95)、最大 1,000 epoch、验证 MSE 早停 patience=10。测试集只在模型选定后评测，不能用来选 epoch。

## 3. 检查点与续训

| 文件 | 用途 |
| --- | --- |
| `latest.pt` | epoch 边界续训，含最新模型、AdamW、StepLR、早停计数、最佳模型副本及各 rank Torch/NPU RNG |
| `checkpoint.pt` | 验证集选出的最佳模型，只用于独立推理验证，不含可继续训练的优化器状态 |
| `epoch_history.json` | 每个 epoch 的损失、学习率、训练样本数、早停计数和最佳 epoch |
| `train_result.json` | 实际完成范围、停止原因、检查点 SHA、完整验证与独立测试指标 |

续训使用尚未完成训练的 `latest.pt`。保持数据摘要、rank 数、全局 batch、lr 和确定性配置相同。

```bash
# 第一次：完成一个 epoch 后主动暂停，保留原输出
bash test/train_full_2p.sh --devices 0,2 \
  --output_dir "$PWD/test/output/full_epoch1" --pause_after_epochs 1
# 第二次：在新目录恢复完整状态
bash test/train_full_2p.sh --devices 0,2 \
  --resume "$PWD/test/output/full_epoch1/latest.pt" \
  --output_dir "$PWD/test/output/full_resumed"
```

full 入口禁止 `--steps` 截断；短步功能训练使用 `train_2p.sh`。full 训练在计算前启用 `torch.use_deterministic_algorithms(True)`。首次未启用该设置时续训对照失败；启用后重新执行，模型/优化器/调度器/最佳模型/RNG 共 164 个张量逐位相等，最大差为 0。底层具体非确定性算子未定位，不能将其归为已修复的 CANN 内核缺陷。

随后扩展全量数值检查发现，运行时默认允许卷积 HF32 时，完整 test 429,780 个输出中仍有 1 个越过原 CPU 容差。同权同输入关闭 HF32 后，全部 val/test 共 859,560 个输出通过，最大差为 5.72e-6/4.77e-6。新增 `npu_precision.py` 显式关闭卷积与 matmul HF32，记录到 checkpoint，并拒绝跨精度完整续训；已在新目录重新完成全训练与后续门禁。没有用修改容差掩盖该问题。

## 4. 已完成的全量结果

2026-10-18 当前显式 FP32 重跑在用户第二台 A3 机器完成 52 epoch、14,404 次全局优化器更新，early stopping，最佳 epoch=42（11,634 更新）。完整独立 test：MAE=2.3743455411、RMSE=4.1293943097、WMAPE=4.0883945979%。全部 val/test 共 859,560 个输出通过原 CPU/NPU 门槛，最大差分别为 4.5299530e-6 / 4.7683716e-6，越界 0。

第一轮默认 HF32 的历史 test 为 MAE=2.3669233872、RMSE=4.1415581046、WMAPE=4.0756143630%，但全量数值有上述 1 点未过，因此不能替代当前完整 FP32 验收。MAE/RMSE 沿用原始速度数值单位；WMAPE 是 `sum(abs(error))/sum(truth)`，与功能基线 MAPE 的定义不同。

依据：[完整训练及测试](validation/full_training_20261018.json)、[续训控制](validation/full_resume_20261018.json)。这些结果证明当前配置完成了训练和测试流程；没有用户提供的业务精度目标，因此不声明业务精度验收或与论文数值严格等价。

独立离线包已包含完整数据与选出的权重，可先复验，再决定是否重新训练：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

## 5. 独立梯度、吞吐和持续训练

完整训练目录中的 `latest.pt` 可作为以下独立检查的起点。各命令使用新的输出目录；完整预测 CPU/NPU 对照另外比较全部 val/test 输出，而非只比较一批输入。

```bash
python3 tools/validate_extended.py --mode full --check gradient --nproc 1 \
  --devices 0 --checkpoint "$PWD/test/output/full_resumed/latest.pt" \
  --output_dir "$PWD/test/output/full_gradient"
python3 tools/validate_extended.py --mode full --check stability --nproc 2 \
  --devices 0,2 --checkpoint "$PWD/test/output/full_resumed/latest.pt" \
  --duration_seconds 1800 --output_dir "$PWD/test/output/full_stability"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/full_stability"
```

`--check benchmark --nproc 1/2` 分别测量真实加载、传送、前后向、优化器和逐步同步开销。每 rank batch 保持相同，是弱扩展测量；没有生产吞吐目标或加速承诺。梯度控制在 eval 模式保留 autograd，以隔离随机 dropout 和训练 BN；它不能替代真实 train 模式持续训练。持续训练必须实际达到 1,800 秒，损失/梯度/状态有限、参数更新和各 rank 参数 SHA 一致，并通过保存后的独立重载。

本轮完整数据梯度控制：28 个有效梯度张量全部通过，整体相对 L2=8.6354638e-7，一步优化器参数最大差 2.2798777e-6。每 rank batch=16 的 1p / 2p 真实训练吞吐分别为 1,282.998 / 1,692.274 窗口每秒。双卡稳定性实际运行 1,800.139611 秒、86,190 次更新；独立重载 CPU/NPU 最大差 2.1457672e-6。

持续训练循环用于检查稳定性，不能替代原完整训练的选模、调度和早停，也不能使用其检查点继续原完整协议；继续正式训练应使用完整训练目录的 `latest.pt`。质量指标始终对应 52 epoch 运行所选 `checkpoint.pt`，不是持续训练后另外改变的权重。
