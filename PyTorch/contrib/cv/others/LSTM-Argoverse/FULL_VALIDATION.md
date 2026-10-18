# LSTM-Argoverse 完整训练与验收

2026-10-18 已在用户提供的第二台 A3 机器完成下述完整数据流程。逐项实测见 [本模型结果](validation/extended_acceptance_20261018.json)。先复验离线包可执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

该命令校验完整包、准备锁定镜像，并在新输出目录复制及重载随包权重。准备与重载均只使用指定的一张物理卡，原封包文件和结果保留。

## 1. 模型和数据

本模型用于车辆 XY 轨迹预测，观察 20 帧、预测 30 帧。编码器、解码器及坐标转换采用 [原始实现](https://github.com/jagjeet-singh/argoverse-forecasting) 的固定源码；模型含 14 个参数张量。完整模式使用 Argoverse v1.1 全部 205,942 条 train 和 39,472 条 val，不以 128/32 功能子集替代。

`full_data_sources.lock.json` 固定原始 train/val 归档的官方地址、字节数和 SHA256。预处理逐条调用原坐标函数，核验每条 50 个真实 AGENT 时刻、原始 CSV 内容去重与跨划分互斥、历史变换不使用未来、坐标逆变换。完整来源清单压缩保存在 `data/full/*_source_sequences.jsonl.gz`。

## 2. 运行

使用 [README](README.md) 中锁定的同一个 ARM64 镜像。解压本模型独立离线包，进入本目录；所有输入和输出保存在本模型中。

```bash
bash test/setup.sh
bash test/prepare_data.sh --mode full
bash test/train_full_2p.sh --devices 0,2
bash test/validate_1p.sh --devices 0
```

已有完整 NPZ 时准备入口检查全部摘要后复用；从原始数据重新准备时，需要 `data/raw/forecasting_train_v1.1.tar.gz` 与 `forecasting_val_v1.1.tar.gz`。缺少归档、样本数不符或 SHA 不同会拒绝执行。full 入口禁止短步 `--steps` 截断；十步检查使用原 `train_2p.sh`。

## 3. 原始 curriculum 与分布式适配

| 项目 | 完整模式 |
| --- | --- |
| 训练 rollout | 依次 1、10、30 步 |
| 优化器 | encoder Adam 与 decoder Adam 独立保存状态，lr=0.001 |
| batch | 全局 512；两卡每 rank 256 |
| epoch 上限 | 原始全局 5,000 epoch，不是每阶段 5,000 |
| 验证 | 每 5 epoch 使用全 39,472 条 val，预测完整 30 步 |
| 验证 loss 聚合 | 与原始实现一致，按各 batch 的 MSE 均值再平均，保留最后短 batch |
| 阶段推进 | 连续三次验证 loss 比上一次增大后进入下一 rollout |
| 末批处理 | 每个 epoch 覆盖全部训练轨迹，按各 rank 实际样本数加权；零权重补位不计入样本或 loss |
| 选模 | 保存各阶段最佳模型，正式输出优先选择 30 步阶段最佳权重 |

训练保留原生 `nn.LSTMCell`。NPU batch=1 的兼容分支只复制一条独立输入补成 batch=2，并在 loss/指标前裁去补齐行；默认完整 batch 不使用此分支。完整训练没有改成另一个 LSTM 网络。

## 4. 续训和完成判断

```bash
bash test/train_full_2p.sh --devices 0,2 \
  --output_dir "$PWD/test/output/full_epoch1" --pause_after_epochs 1
bash test/train_full_2p.sh --devices 0,2 \
  --resume "$PWD/test/output/full_epoch1/latest.pt" \
  --output_dir "$PWD/test/output/full_resumed"
```

`latest.pt` 含最新模型、两个 Adam、epoch/update/stage、阶段早停计数、各阶段最佳模型和所有 rank Torch/NPU RNG；续训要求同一数据摘要、rank 数、全局 batch 和 lr。`checkpoint.pt` 是选择后的推理权重，不能替代 `latest.pt` 继续训练。

已完成“1 epoch 暂停＋恢复至 2 epoch”与“连续 2 epoch”对照：806 更新，60 个模型/两个优化器/阶段/RNG 张量全部逐位相同，最大差 0。另从第 689 epoch 的成熟状态快照恢复，跨过第 690 epoch 的全量验证，与连续训练至 691 epoch 对照：包含非空的三个阶段最佳权重，共 102 个张量全部逐位相同、最大差 0；原长训练未被中断。

原始三阶段长训练实际完成 725 epoch / 292,175 次更新，各阶段分别 165 / 150 / 410 epoch，按原连续三次验证增大规则结束。选择第 535 epoch 的 30 步阶段最佳模型，随后独立重载和全量数值检查通过。没有缩小 5,000 epoch 上限来提前结束验证。

只有 `full_reference_protocol_completed=true` 才能声明本项目默认三阶段完整配置已完成。`global_epoch_limit` 提前到达而阶段没有覆盖、只主动暂停、没有 30 步阶段选优，均不能宣称完整 curriculum 验收。

## 5. 独立验证和指标

完整训练后的 `validate_1p.sh` 在独立容器严格重载，核对检查点 SHA，重放保存参考，比较同权 CPU/NPU，并对全部 39,472 条 val、2,368,320 个未来坐标元素做数值控制。`full_prediction_control.json` 记录全部覆盖数、最大误差、越界数及两个后端恢复到 city 坐标后的 ADE/FDE/Miss Rate。

预测坐标沿用原数值阈值 rtol=atol=0.003；ADE/FDE 单位米，Miss Rate 阈值 2 米、K=1。此 val 同时用于选模，不能说成不参与模型选择的独立 test 或 Argoverse 排行榜成绩。实际训练和指标通过也不代表获得驾驶业务准入。

| 实测项目 | 当前结果 |
| --- | --- |
| 全部验证坐标 | 2,368,320 个，越界 0；最大绝对差 4.25055623e-6 |
| 同批次 CPU/NPU city 指标 | batch512；ADE 1.70117643 m / 1.70117643 m，FDE 3.87356091 m / 3.87356088 m；MR 均 0.6511197811 |
| 最终完整权重梯度控制 | 真实 batch512，原两个独立 Adam；14 个有效梯度全部通过，相对 L2 1.54778409e-6 |
| 一步优化器参数对照 | 最大差 1.41561031e-7，通过 |
| 选定推理权重 SHA256 | `306ca75daba7116ed6e4cc2dc93df5c20b3ea08d6fe04995072fc957146ff29a` |
| 最新续训状态 SHA256 | `6cd47a245016d786a5ec971851e2852bd8d566ad91643e09aa81d9773172bb91` |

长稳覆盖来自超过 30 分钟的实际完整训练。独立 1p/2p 吞吐微基准使用 128/32 功能数据、每 rank batch8，与完整训练全局 batch512 分别报告，不能混作完整数据性能。没有给定驾驶业务 ADE/FDE/MR 或生产吞吐验收门槛。

## 6. 保留和复现

每次训练使用新输出目录；不要覆盖失败或旧基线。保留 `latest.pt`、`checkpoint.pt`、`epoch_history.json`、各 rank 结果、独立验证 JSON 和原始数据 SHA。镜像、数据及代码版本变化后应重新做恢复与数值控制。Atlas 800I A3 是推理系列，本项目真实实验不改变官方训练支持范围。

完整原始 CSV 已在同一锁定 ARM64 镜像中重新预处理，245,414 条来源逐条一致；数组在预设 f32/f64 容差内相同，city truth/reference/translation 逐位相同。重建使用新目录，没有替换长训练绑定的数据摘要；具体环境、最大差和全部数组结果在本模型 JSON 中保留。
