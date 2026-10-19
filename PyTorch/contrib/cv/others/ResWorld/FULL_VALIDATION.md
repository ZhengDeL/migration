# ResWorld 独立迁移与验证指南

## 1. 当前验收范围

2026-10-18在7.6.28.252的固定A3环境完成本次数值修复及独立回归。当前结果为 `pass_mini_numerical_and_operational_gates_full_protocol_pending`；逐项实测见[JSON](validation/extended_acceptance_20261018.json)。

| 检查 | 实际结果 |
| --- | --- |
| 原bf7a失败、后续07a6失败、旧72afe、新训练20步、最终长稳五份权重 | 每份446有效梯度全部通过原逐参数、整体L2、loss和一步参数门槛 |
| 常量几何修改前后CPU参考 | 446/446完整梯度逐位相同 |
| 四卡10+10恢复 vs连续20 | 2054/2054状态张量逐位相同、max0；全部rank增强后真实输入一致 |
| 四卡持续训练 | 1811.967955秒、660更新；模型/优化器有限、参数更新、rank参数SHA一致 |
| 最终权重独立梯度 / 重载 | pass；relativeL2=2.83294570892e-05；独立重载通过 |
| 作者完整权重原65帧 | CPU/NPU轨迹和逐帧原离散碰撞对照通过 |
| 正式trainval/12epoch和驾驶业务精度 | 尚未完成 |

## 2. 完整离线包一条命令复验

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

完整解压ResWorld.zip，在本README目录执行。该入口检查包内清单和固定镜像，将随包最终checkpoint、参考和训练记录复制到新目录后验证重载。它是重载入口；完整数值门禁见第4节。GitHub源码目录不含大镜像、数据或交付权重。

## 3. 上游、环境与逐文件适配

作者源码固定 `a5ae973001dd6001d95b3389020a9e263d73a56e`，发布前核对仍为HEAD。镜像和精确image ID、版本见本目录README与images.lock.json。沿用原80766238参数、704 state、3-command×6-step输出与三项原loss。

真实mini含323/81标注帧、8/2互斥场景、20个真实CAN文件和2424个投影深度文件；验证使用全部65个完整future有效帧。

| 文件 | 实际修改 |
| --- | --- |
| npu_activations.py | 仅原三个GELU使用官方NPU erf接口，CPU默认erf保持；执行计数、数值profile校验 |
| temporal_residual.py | CPU/NPU以S@(F0-F1)等价替换S@F0-S@F1，保留重复位置量零导数连接 |
| camera_geometry.py | 原get_sample_coor与原归一化固定CPU FP32；仅常量几何，拒绝可学习输入；网格传回NPU |
| prepare_source.py | vendor字节/SHA保持，只在runtime副本打可追踪补丁；文件锁保护多rank准备 |
| npu_compat.py | 原可训练池化、原MMCV NPU autograd；no_grad保留DrivingSDK；CPU add/relu公式兼容 |
| npu_precision.py / deterministic_communication.py | 关闭HF32；HCCL确定性与同步mean allreduce |
| training_state.py / train.py / resume_control.py | 全rank RNG及游标、增强前跳过已消费批次、输入SHA与三个数值profile |

相机标定原FP64矩阵计算和固定相机投影属于无梯度元数据处理；NN参数、特征、loss、有效梯度和采样反向均在NPU。Torch基础wheel的`+cpu`版本标签不代表NN在CPU执行，实际设备和调用守卫已实测。

## 4. 数值修复与五份固定权重回归

```bash
bash test/validate_numerical_ops_1p.sh --devices 0
python3 tools/validate_extended.py --check gradient --devices 0 \
  --checkpoint "$PWD/test/output/reference_weights/original_failure/checkpoint.pt" \
  --output_dir "$PWD/test/output/original_case_recheck_new"
python3 tools/validate_extended.py --check gradient --devices 0 \
  --checkpoint "$PWD/test/output/reference_weights/new20_failure/checkpoint.pt" \
  --output_dir "$PWD/test/output/new20_original_case_recheck_new"
python3 tools/validate_extended.py --check gradient --devices 0 \
  --checkpoint "$PWD/test/output/reference_weights/old_stability/checkpoint.pt" \
  --output_dir "$PWD/test/output/historical_stability_recheck_new"
python3 tools/validate_extended.py --check gradient --devices 0 \
  --checkpoint "$PWD/test/output/geometry32_v7_20261018/resumed20/checkpoint.pt" \
  --output_dir "$PWD/test/output/new_training_gradient_recheck_new"
python3 tools/validate_extended.py --check gradient --devices 0 \
  --checkpoint "$PWD/test/output/geometry32_v7_20261018/stability/checkpoint.pt" \
  --output_dir "$PWD/test/output/final_gradient_recheck_new"
```

算子检查独立覆盖官方GELU前后向对原公式FP64、稳定残差三种输入梯度对原公式FP64、原CPU FP32几何逐位一致，以及NN采样反向留NPU和可学习元数据拒绝。它不能替代五份真实网络检查。

整网使用同权同真实批次、eval保留autograd，逐参数maxdiff≤.001+.01×CPU refmax，整体relativeL2≤.01；loss rtol/atol=.003，一步参数rtol=.003、atol=.0003。门槛未放宽，不是华为业务标准。一步AdamW归一化会压缩差异，不能替代梯度门禁。

## 5. 四卡从头训练和严格恢复

```bash
bash test/setup.sh
bash test/train_4p.sh --devices 0,2,4,6 --steps 10 --output_dir "$PWD/test/output/first10_new"
bash test/train_4p.sh --devices 0,2,4,6 --steps 20 \
  --resume "$PWD/test/output/first10_new/checkpoint.pt" --output_dir "$PWD/test/output/resumed20_new"
bash test/train_4p.sh --devices 0,2,4,6 --steps 20 --output_dir "$PWD/test/output/continuous20_new"
python3 tools/container_resume_control.py "$PWD/test/output/resumed20_new/checkpoint.pt" \
  "$PWD/test/output/continuous20_new/checkpoint.pt" "$PWD/test/output/resumed20_new/resume_control.json"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/resumed20_new"
```

每rank batch1、AdamW1e-4/wd.01、clip35。steps表示累计目标；20不是恢复后再加20。实测四卡逻辑4,6,12,14，每物理卡仅选一个逻辑ID。schema2校验rank、数据、batch、lr、精度、MSDA、HCCL和三个数值profile，恢复全rank RNG及下一批游标；旧profile会拒绝精确续训。

## 6. 吞吐、30分钟和最终重载

```bash
python3 tools/validate_extended.py --check benchmark --nproc 4 --devices 0,2,4,6 \
  --checkpoint "$PWD/test/output/geometry32_v7_20261018/resumed20/checkpoint.pt" --output_dir "$PWD/test/output/benchmark4p_new"
python3 tools/validate_extended.py --check stability --nproc 4 --devices 0,2,4,6 \
  --checkpoint "$PWD/test/output/geometry32_v7_20261018/resumed20/checkpoint.pt" --duration_seconds 1800 \
  --output_dir "$PWD/test/output/stability_new"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/stability_new"
```

单卡计时将nproc设1、devices设单ID。本次1/4卡吞吐为0.375813174/1.20601454规划帧/秒；一帧含三时刻六相机，不能称单幅图像FPS。计时含真实加载、传送、前后向、优化器及同步；共享宿主无生产性能目标。

长稳最终权重SHA256：`24e405e0f15156e7618a697be665d95bc53e5ef856431e2a426ae4287a505d3f`。长稳权重用于重载；精确续训使用含全部RNG/游标的schema2训练checkpoint。rank参数SHA一致不表示BN buffer逐位一致。

## 7. 原失败、根因和未采用候选

详见[数值修复说明](NUMERICAL_REPAIR.md)与带日期[失败历史](validation/history/pre_numerical_fix_20261018.json)。普通GELU公式差异、长归约后相减及常量几何投影舍入是本次已证实的三条路径。稳定残差改变CPU参考的浮点顺序，已用未修改原公式FP64规划头独立核对；常量几何这一项则有446个CPU参考梯度逐位不变的直接证明。

没有部署捕获ReLU masks、修改ReLU导数或NN CPU反向。GELU单独、仅NPU稳定残差对旧CPU、高精度配置、内部格式、补偿卷积和全网/几何FP64等未通过候选保留为历史。旧JIT出现不可信输出后被拒绝采用。

## 8. 作者原65帧及正式协议边界

```bash
bash test/validate_planning_1p.sh --devices 0 --checkpoint "$PWD/ckpts/resworld-official-final.pth" \
  --output_dir "$PWD/test/output/official_planning_recheck_new"
```

704 state strict load，原ResWorld.compute_planner_metric_stp3；全部65帧预测rtol=atol=.003、逐帧原离散碰撞值一致。每后端独立复制GT并确认原标注未变。作者预训练是否覆盖mini未排除，导航command来自标注，不能据此宣称独立论文复现或驾驶业务精度。

正式trainval/CAN/地图/深度、原12epoch/GeoBEV初始化、EMA10560/.999、SyncBN epoch0、linear warmup500/1⁄3、AdamW1e-4/wd.01、clip35、StepLR[24]与完整评测仍需完成。full入口继续明确拒绝；不能只补全量路径便称完整训练通过。Atlas800I A3为推理系列，实测不改变官方训练支持范围。

## 9. GeoBEV、SyncBN、EMA 与 warmup 的四卡预检

在完整训练之前，单独执行原仓训练组件的真实网络预检：

```bash
bash test/preflight_official_protocol_4p.sh --devices 0,2,4,6 \
  --steps 8 --output_dir "$PWD/test/output/original_protocol_preflight_new"
```

使用本模型固定镜像、已有真实 mini、真实 CAN 和投影深度，以及 `ckpts/geobev-r50-nuimage-cbgs.pth`。选择四张空闲物理卡，每张只选一个逻辑设备。指定全新的输出目录；不修改原功能训练的 `latest_run.json`。GitHub 源码目录本身不包含上述大文件。

| 原仓组件 | 新预检入口实际调用 |
| --- | --- |
| GeoBEV 初始化 | 原 MMCV `load_checkpoint(strict=False)`；逐个确认所有名称与形状匹配的张量确实载入，完整保存 missing / unexpected keys |
| SyncBN | 原 `SyncbnControlHook` 使用的 `torch.nn.SyncBatchNorm.convert_sync_batchnorm`；从 epoch 0 启用，记录真实 NPU 前向次数 |
| EMA | 直接调用固定作者源码 `ModelEMA`，使用 `MEGVIIEMAHook` 的 decay=0.9990 与 init_updates=10560 |
| 学习率 | 直接调用 MMCV `StepLrUpdaterHook`；linear warmup=500、ratio=1/3、step=[24]，检查 0 / 499 / 500 次更新及 epoch 24 边界 |
| 优化器 | 原 AdamW lr=1e-4、weight_decay=0.01；梯度裁剪 max_norm=35 |
| 批次与卡数 | 每卡 2 帧、四卡合计 8 帧；本入口只执行 2～32 次更新 |

依据：[固定原配置](vendor/ResWorld/projects/configs/resworld/resworld_config.py)、[作者 EMA 实现](vendor/ResWorld/projects/mmdet3d_plugin/resworld/hooks/ema.py)、[作者 SyncBN 实现](vendor/ResWorld/projects/mmdet3d_plugin/resworld/hooks/syncbncontrol.py)。沿用本模型已经验证的 FP32、MSDA autograd、几何和时序数值适配。

成功必须同时满足：执行退出 0、四个 rank 的全部更新完成、原三项 loss 与训练/EMA/优化器状态有限、真实执行 SyncBN、全 rank 参数摘要一致、模型和 EMA 保存后重读逐位相同。结果为输出目录中的 `protocol_preflight.json`，判定值 `pass_geobev_syncbn_ema_warmup_four_card_mini_preflight`。`command.json`、`console.log`、`execution.json` 和四份 `rank_*.json` 保留命令、退出码与逐步数据。

此项使用现有 mini 功能训练帧，检查点是预检格式，不用于声明完整训练或精确续训。正式数据派生、原 12 epoch runner、完整训练/EMA 恢复、最终规划与碰撞评测仍须单独验收；现有 `full` 数据门禁保留。

2026-10-19 实测：四卡各 8 更新、64 个不同真实训练帧；548 个 GeoBEV 兼容张量核对载入，86 个 SyncBN 模块每 rank 实际调用 1,824 次；EMA updates=10,568，保存后重读逐位相同；全部命令退出 0。四 rank 第一步 / 第八步 loss 均值分别为 10.3489307165 / 12.0367364883，仅记录八步输入和训练行为，不作收敛结论。GeoBEV 的 156 个未匹配模型键和 60 个 checkpoint 额外键完整保留，遵循原 `strict=False` 部分初始化。详见[本次原始逐步数据、边界检查与摘要](validation/original_protocol_preflight_20261019.json)。

## 10. 原 runner 的完整 mini 轮次与 EMA 续训

本节为 2026-10-19 GitHub 新增入口，10 月 18 日网盘 ZIP 尚未重打包。复验时以当前 GitHub 的本模型源码目录建立新工作目录，再复制旧模型包中的 `data/`、`ckpts/`、`images/` 三类资产。新目录保留当前源码和来源锁，不继承旧 `package_manifest.json`：该清单校验的是旧包全部字节。完整验证后另行生成独立包及 SHA；旧目录和原检查点继续保留。

此入口使用原 MMCV `EpochBasedRunner`、镜像内华为 `NPUDistributedDataParallel`、作者 `DistributedGroupSampler` 和原三项训练 hook。保留全部 323 条 mini 训练标注，不使用前述功能训练帧筛选。每卡 batch=2，四卡每轮 41 个 batch；原分组采样器会补齐少量样本，不能把 328 次样本呈现写成 328 个不同标注。

依次执行，每条命令使用新的模型内输出目录：

```bash
bash test/train_native_mini_4p.sh --devices 0,2,4,6 --workers 0 --epochs 1 \
  --output_dir "$PWD/test/output/native_first_new"
bash test/train_native_mini_4p.sh --devices 0,2,4,6 --workers 0 --epochs 2 \
  --resume "$PWD/test/output/native_first_new/exact_epoch_1.pth" \
  --output_dir "$PWD/test/output/native_resumed_new"
bash test/train_native_mini_4p.sh --devices 0,2,4,6 --workers 0 --epochs 2 \
  --output_dir "$PWD/test/output/native_continuous_new"
python3 tools/container_native_epochs.py \
  --compare "$PWD/test/output/native_resumed_new/exact_epoch_2.pth" \
            "$PWD/test/output/native_continuous_new/exact_epoch_2.pth" \
  --output_dir "$PWD/test/output/native_compare_new"
```

`--epochs 2` 表示累计达到两轮；恢复后的本次执行只训练第二轮。比较命令自动在固定镜像中运行 CPU 检查，不要求宿主安装 Torch，不占 NPU。选卡前先确认上一容器已经释放；宿主逻辑 ID 必须属于四张不同物理卡。

原 `epoch_N.pth` 和 `epoch_N_ema.pth` 保留。精确恢复使用新增 `exact_epoch_N.pth`：完整模型、原优化器、EMA 状态和更新计数、原 runner epoch/iter、全 rank Python/NumPy/Torch/NPU RNG 及 DataLoader generator 一起保存。仅支持轮次边界；不将其称为任意 batch 的精确续训。

恢复时先调用原 runner 的 `resume`，再恢复原 EMA；所有原 epoch hook 执行后、创建新迭代器前恢复对应 rank 的 RNG 和 loader generator。数据 metadata 摘要、上游来源锁、GeoBEV 权重、卡数、精度、通信、batch、seed 与 worker 数必须相同。固定镜像由宿主入口核对，执行脚本 SHA 另列在实测证据中；修改脚本后的恢复不属于本记录已验证范围。GeoBEV 在启动 NPU 容器前按 `pretrained_sources.lock.json` 核对。该控制使用 `workers=0`；原配置 `workers=8` 的完整训练和精确恢复不因本测试被认定通过。

本入口另外显式设置 `find_unused_parameters=True`，使 DDP 使用固定构造分桶；这是迁移适配，不是作者配置的原值。原值未设置该选项，续训对照出现了重启后漂移。Torch 2.7.1 的[原始 reducer 实现](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/csrc/distributed/c10d/reducer.hpp#L128-L133)说明该选项如何控制重建分桶。同步 mean-allreduce 的公式不变，实际每步分桶大小和参数顺序摘要写入 `rank_*.jsonl`，与输入、loss 一起比较。该设置可能增加图遍历开销，不据此声明性能提升；新旧分桶 profile 不允许混用精确恢复。

通过判据：三次训练退出 0；恢复/连续两轮的模型、EMA、优化器、游标与全 rank RNG 全部逐位相同；四 rank 第二轮的实际输入摘要、loss、学习率和 EMA 更新轨迹逐条相同。比较报告为 `native_resume_control.json`，失败保留具体字段，不放宽为误差近似。

本入口限制为一至两轮真实 mini，用于验证原 runner 和恢复，不是正式 12 epoch 或驾驶精度验收。原数值门槛与正式 `full` 训练门禁保留。

### 2026-10-19 四卡实际结果

| 检查 | 实测 |
| --- | --- |
| 完整 mini 轮次 | 保留 323 条标注；原组采样器每轮 328 次呈现、每 rank 41 次更新 |
| 一轮后重启续训 | 第一进程 41 次更新；新进程再 41 次，累计 82；两次退出码均为 0 |
| 独立连续两轮 | 82 次更新，退出码 0；EMA updates=10,642 |
| 完整状态 | 模型、优化器、EMA、各 rank RNG 与 loader generator 共 2,762 个张量，全部逐位相同 |
| 第二轮实际轨迹 | 四 rank 共 164 条输入 SHA、loss、lr、EMA 与梯度分桶记录全部相同 |
| 公开比较入口 | 宿主 `container_native_epochs.py --compare` 再次退出 0；CPU 容器未映射 NPU |
| 最终检查点 | 两份文件 SHA256 同为 `6efe38550d7a3688fb75168990e5375da0b89ee1dfe69458ac2d5cd704748ac4` |
| 未覆盖 | 正式 trainval、原 12 epoch、workers=8 的恢复、任意 batch 恢复、最终任务精度 |

[完整命令、时刻、退出码、逐步记录与源码 SHA](validation/native_epoch_resume_20261019.json)。原 `find_unused_parameters=False` 对照的 2,282 项差异与定位依据保留在[失败对照](validation/history/native_epoch_resume_before_fixed_buckets_20261019.json)，没有用近似误差替代逐位判据。

## 11. 正式 trainval 输入准备

使用本模型当前 GitHub 源码中的 `test/prepare_full_data.sh`；旧网盘包尚未包含本入口。

将完整原始数据放到本模型 `data/nuscenes_full/`：

```text
data/nuscenes_full/
  samples/                    # 完整官方相机/点云等原始文件
  sweeps/
  v1.0-trainval/               # 含完整 metadata 与 lidarseg.json
  lidarseg/v1.0-trainval/
  can_bus/
  maps/expansion/             # 四个官方地图 JSON
```

官方 map expansion ZIP 的顶层目录是 `expansion/`，解压位置应使地图落在 `data/nuscenes_full/maps/expansion/`；仅把它解到数据根目录不符合作者地图读取路径。数据必须实际位于本模型内。

```bash
bash test/prepare_full_data.sh --data_path "$PWD/data/nuscenes_full"
```

该命令不映射 NPU。在固定镜像内先检查完整 850 场景、34,149 样本、官方场景划分、全部 metadata 引用的传感器文件、完整 lidarseg、地图和 CAN，再调用固定作者的 `vad_nuscenes_converter.py`（trainval、max_sweeps=10）与 `generate_point_label.py`。不会用 mini 标注补成 trainval。

固定镜像的模型算子包在导入时要求可用 NPU；数据准备改由 `cpu_data_converter.py` 隔离原转换器的两项模型依赖：从镜像原 `NuScenesDataset` 源码提取同一 `NameMapping`，其余转换算法直接执行原作者代码。转换器导入时的 CLI 解析节点也单独隔离，并校验原函数 AST 保持。仅用于单独 2D 导出的 `points_cam2img` 在本入口明确拒绝调用。本入口只做元数据、NumPy 与深度投影计算。

有真实 mini 的工程师可先做独立 CPU 转换对照：

```bash
bash test/prepare_full_data.sh --check-mini-components \
  --data_path "$PWD/data/nuscenes" \
  --output_dir "$PWD/test/output/cpu_data_converter_control_new"
```

该控制在新模型内目录中准备原始 mini 的普通文件硬链接，调用原 mini 转换（max_sweeps=0）和原六相机深度生成，逐项比较现有原标注的数组、其他字段及全部 2,424 个深度文件。只归一化传感器根目录位置，其他内容要求相同。输出 `cpu_conversion_control.json`；它检验 CPU 导入隔离和 mini 转换，不能代替完整 trainval 的 max_sweeps=10 转换。

完整输入要求为 28,130 train / 6,019 val、700 / 150 场景、204,894 个六相机投影深度文件。转换后逐个核对 token、场景、路径和深度格式，并生成所有输入的流式 SHA256 清单 `input_manifest.jsonl` 与 `metadata.json`。已存在的完整标注或深度会经过检查再复用；本次是否生成分别记录在 `annotations_generated_this_invocation` / `depth_generated_this_invocation`，不把复用文件说成本次生成。再次执行会复核已准备输入的摘要。原作者对于确实缺少 CAN 的记录所采用的 fallback 予以保留并计数，结果写在 `official_can_fallback_samples`，不生成项目自造 CAN 值。

2026-10-19 已实测三个负向用例：真实 mini、仅改名为 trainval 的真实 mini、模型目录外路径均被拒绝，原 mini metadata 未改。公开 CPU 转换控制退出 0：323/81 帧、14,140 个标注数组及其余字段相同，2,424 个深度文件相同，NPU 设备未映射；原转换函数 AST 保持。完整命令、来源摘要、原导入/CLI 失败与修复后结果见[实测记录](validation/full_preparation_guards_20261019.json)。完整传感器下载仍在进行，本入口的正向完整转换、原 12 epoch 训练及全量精度尚未完成。数据准备成功也不能替代训练/评测验收。
