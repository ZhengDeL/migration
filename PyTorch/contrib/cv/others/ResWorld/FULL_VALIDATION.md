# ResWorld 独立迁移、验证与当前限制

## 1. 先看结论

2026-10-18 在用户新增 A3 机器实际执行完整 ResWorld 网络的真实 mini 四卡训练、严格续训、数值、吞吐、30分钟与独立重载；另严格加载作者完整权重，完成原65帧规划/碰撞CPU/NPU控制。逐项结果见[本模型实测](validation/extended_acceptance_20261018.json)。

| 检查 | 当前配置结果 |
| --- | --- |
| 四卡10＋10 vs 连续20步 | 2054/2054状态张量逐位相同，最大差0；全部rank实际增强输入一致 |
| CPU/NPU梯度 | **fail**；446个有效梯度，156个参数未过门槛，整体相对L2=0.0264092168708 |
| loss与优化器一步 | loss接近=True；一步参数控制=True；不能覆盖梯度失败 |
| 四卡持续训练 | 1806.233653秒，630次更新；有限状态、实际更新、各rank参数SHA一致 |
| 长训练独立重载 | pass_checkpoint_reload_and_validation |
| 作者完整权重原65帧评测 | pass_original_mini_planning_collision_control；逐帧原碰撞值一致 |
| 正式nuScenes trainval/12epoch协议 | 未完成；full入口保持拒绝 |

**当前整体数值验收未通过。** 严格续训的漂移已在新配置处理并复验；单卡CPU/NPU梯度失败仍需定位。该单卡检查没有执行HCCL梯度归约，不能直接把其残留归因于四卡通信。没有放宽事先声明的门槛。

## 2. 独立离线包：一条命令复验

完整解压本模型 ZIP，进入含本 README 的目录，在 ARM64 Linux Ascend 服务器运行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

`--devices`填写一个空闲的宿主逻辑NPU ID。该入口核验整包文件清单、准备精确固定镜像，再把随包检查点/参考/训练记录复制到新输出目录完成重载。原封包记录保留；输出会给出新结果位置。镜像、宿主依赖和image ID见[README](README.md)及`images.lock.json`。GitHub源码目录不包含大镜像、数据或交付检查点。

## 3. 模型、真实数据与逐项适配

采用[作者ResWorld](https://github.com/mengtan00/ResWorld)固定提交`a5ae973001dd6001d95b3389020a9e263d73a56e`；保留80766238参数、704个state张量、三时刻×六相机、原depth和两项规划loss、3-command×6-step输出。真实mini为323/81标注帧、8/2互斥场景，20个真实CAN消息文件、2424个投影深度文件，验证覆盖全部65个完整future有效帧。

| 位置 | 当前迁移处理 |
| --- | --- |
| `prepare_source.py` | vendor保持锁定字节，只在runtime副本做可追踪替换；fcntl排他锁保护多rank源码/manifest准备 |
| `npu_compat.py` | 新训练autograd使用原MMCV公式与原生NPU grid_sample；no_grad推理保留DrivingSDK公共API；记录实际调用并做非零守卫 |
| ResNet池化 | 原3×3/stride2/pad1可训练ATen max-pool，替代镜像只有前向的融合池化 |
| 标定 | 无梯度矩阵按原FP64公式在CPU计算，再传回FP32 NPU；不把NN反向移到CPU |
| 数据传送/CPU对照 | 显式传送实际张量；CPU的fused add/relu恢复等价Torch公式 |
| `npu_precision.py` | 显式关闭conv/matmul/aclnn HF32，保存/恢复实际配置 |
| `training_state.py` | 下一epoch/batch游标、全部rank Python/NumPy/Torch/NPU RNG；图像增强前跳过已消费批次；实际输入SHA |
| `deterministic_communication.py` | 创建首个HCCL通信域前启用确定性；公开DDP hook在NPU同步mean allreduce，显式除world_size，记录实际调用 |

原生NPU `grid_sampler_2d`及其backward的PrivateUse1内核注册已实测；训练检查参数、loss和有效梯度为NPU张量。同步归约保留DDP平均梯度口径，减少通信/计算重叠；性能使用该新配置重新计时，不沿用旧异步配置吞吐。

## 4. 新四卡功能训练与严格恢复

```bash
bash test/setup.sh
bash test/train_4p.sh --devices 0,2,4,6 --steps 10 \
  --output_dir "$PWD/test/output/first10_new"
bash test/train_4p.sh --devices 0,2,4,6 --steps 20 \
  --resume "$PWD/test/output/first10_new/checkpoint.pt" \
  --output_dir "$PWD/test/output/resumed20_new"
bash test/train_4p.sh --devices 0,2,4,6 --steps 20 \
  --output_dir "$PWD/test/output/continuous20_new"
python3 tools/container_resume_control.py \
  "$PWD/test/output/resumed20_new/checkpoint.pt" \
  "$PWD/test/output/continuous20_new/checkpoint.pt" \
  "$PWD/test/output/resumed20_new/resume_control.json"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/resumed20_new"
```

每rank batch1、AdamW1e-4/weight decay0.01、clip35。steps表示目标累计更新数；20为累计20步，不能理解为恢复后再增加20步。新schema2严格续训核对数据、rank数、batch、lr、FP32、MSDA和通信profile；旧配置文件不能直接作为新配置的严格恢复。

本轮实测宿主逻辑4,6,12,14，对应物理2,3,6,7。命令中的0,2,4,6是四张不同物理卡的示例，可改为空闲卡；不要用同物理卡的两个逻辑ID充当两卡。SHA用于确定正在读取哪份文件，新运行的逻辑恢复一致性以完整状态对照JSON判定。

## 5. 梯度、吞吐与30分钟：明确源检查点

随包当前控制的源权重为`test/output/reference_autograd_v3_sync_fp32_20261018/first10/checkpoint.pt`，SHA：`bf7a176b0827bdc88f9fb0d71da3280dd1de73317a3a9686372765257f1ba4e5`。不要省略checkpoint参数后误用历史`delivery_verified`权重。

```bash
python3 tools/validate_extended.py --check gradient --nproc 1 --devices 0 \
  --checkpoint "$PWD/test/output/reference_autograd_v3_sync_fp32_20261018/first10/checkpoint.pt" \
  --output_dir "$PWD/test/output/gradient_recheck_new"
python3 tools/validate_extended.py --check benchmark --nproc 4 --devices 0,2,4,6 \
  --checkpoint "$PWD/test/output/reference_autograd_v3_sync_fp32_20261018/first10/checkpoint.pt" \
  --output_dir "$PWD/test/output/benchmark4p_new"
python3 tools/validate_extended.py --check stability --nproc 4 --devices 0,2,4,6 \
  --checkpoint "$PWD/test/output/reference_autograd_v3_sync_fp32_20261018/first10/checkpoint.pt" \
  --duration_seconds 1800 --output_dir "$PWD/test/output/stability_new"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/stability_new"
```

梯度控制使用同权同批真实数据、eval模式保留autograd，隔离随机dropout和训练BN。每参数张量最大梯度差≤0.001＋0.01×CPU该参数最大梯度，整体相对L2≤0.01；一步参数rtol0.003/atol0.0003，loss rtol=atol0.003。它是预先声明的项目数值门槛，不是华为业务标准。当前gradient返回失败是已知真实结果，保留JSON和日志，不能把一步参数接近替代梯度通过。该一步使用空AdamW初态，归一化效应会压缩部分梯度差；当前最大参数差约2e-4，小于参数atol3e-4，不能单靠该项证明反向正确。全部参数的实际判定见`validation/gradient_parameters_20261018.json`。

吞吐`--nproc 1`和`--nproc 4`分别测量；一个样本是三时刻六相机的规划帧，不是单幅图像FPS。计时包括真实加载、传送、前后向、优化器与同步；CPU/存储与其他任务共享，无生产吞吐目标。稳定性检查点含模型与优化器，不含精确续训的全部游标/RNG；精确恢复使用上节schema2训练检查点。

## 6. 失败、修复与结论范围

- 原DrivingSDK MSDA真实输入重复前向一致，但所测value反向有2.33e-10～4.66e-10差异；不据此宣称解释全部端到端误差或定位到内核行。
- 同一物理卡、相同输入/初始化/RNG下，SDK的单卡20步状态重放失败；原MMCV NPU公式2045/2045逐位相同。仅替换注意力不能消除默认通信下的四卡漂移。
- 只拦截MMCV Function.apply的早期尝试没有覆盖镜像直接调用SDK公共API；后续直接API拦截和执行计数解决了路由验证缺口。
- 两卡只开启HCCL确定性后最大状态差由0.48760降至4.81345e-5，仍失败；同步平均归约的四卡控制2054/2054逐位相同，公开训练入口整合后再次逐位相同。
- 当前单卡CPU/NPU梯度仍失败，说明严格恢复修复与CPU/NPU梯度一致性是独立门禁。旧权重loss/特征/规划头隔离观察到内部非光滑分支差异，但尚未证明当前所有误差的唯一根因；没有用改网络、CPU训练反向或放宽门槛判通过。
- 当前同权/同输入三组对照中，原配置、显式must_keep_origin_dtype、再加high_precision均得到相同2.6409%误差及156个失败参数，没有改善。原配置查询空字符串不能被写成低精度模式。实际MHA入口CPU/NPU各7次、need_weights均True，PyTorch SDPA入口两端均0；不把未执行的SDPA入口当成根因。这些诊断选项没有作为修复加入当前训练配置。

同步hook遵循[PyTorch公开DDP接口](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)。HCCL确定性用途参考[华为官方精度定位案例](https://www.hiascend.com/doc_center/source/zh/mindstudio/80RC1/practicalcases/LargeModelTrainingAccuracy/MindStudio%208.0.RC1%20%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E5%AE%9A%E4%BD%8D%E6%A1%88%E4%BE%8B%2001.pdf)。本项目当前处理效果来自上述实际对照，不是官方保证所有模型逐位确定性的结论。

## 7. 作者权重与原65帧规划/碰撞

`pretrained_sources.lock.json`固定作者README的GeoBEV和完整ResWorld来源、下载后字节/SHA。GeoBEV只有548个同名同形状张量、4个形状不一致项，不是完整ResWorld权重。作者最终模型704个state张量strict load通过，不含优化器。

```bash
bash test/validate_planning_1p.sh --devices 0 \
  --checkpoint "$PWD/ckpts/resworld-official-final.pth" \
  --output_dir "$PWD/test/output/official_planning_recheck_new"
```

使用原`ResWorld.compute_planner_metric_stp3`，全部65帧CPU/NPU预测rtol=atol0.003、逐帧原离散碰撞值完全相同。当前NPU原1/2/3秒平均L2为0.113648963/0.275758918/0.573307600米；原点碰撞、框碰撞及其ST-P3变体在这些帧均为0。碰撞值保持0—1比例，转百分比乘100。

原rasterizer会修改GT boxes的NumPy yaw视图；评测为每后端深复制并检查原标注未变，避免复用已修改GT。no_grad推理保留原SDK路径，新增训练通信hook没有改变该评测链路。作者预训练是否覆盖mini未排除，导航command来自标注；不得将65帧结果当作独立完整验证集、论文质量或驾驶安全验收。

## 8. 正式完整训练仍需完成

需要真实nuScenes trainval/CAN/地图/深度及原标注，完整实现并验证作者12epoch、GeoBEV初始化、AdamW1e-4/wd0.01、clip35、StepLR step[24]、linear warmup500/ratio1/3、EMA init_updates10560/decay0.9990、SyncBN start_epoch0与原规划/碰撞评测。step[24]在12epoch以外，不能擅自改成cosine后声称原配方。

当前`model_task.check_data(..., 'full')`保持明确拒绝；功能脚本未完成上述正式runner/hooks，不能只补入全量数据路径就声称12epoch通过。Atlas800I A3为推理系列，本实验不改变官方训练支持范围。每次新验证使用新目录，保留失败与旧配置文件。
