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
