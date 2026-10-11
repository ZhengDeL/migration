# 1# 快速上手

适用于已经准备好本仓镜像和源码的 1#。所有命令都在服务器的**仓库根目录**运行，一次运行一个模型。

```bash
cd /var/ai-model/lzd/migration
npu-smi info  # 确认要使用的物理卡空闲
```

## 1. 选择模型

| 想运行的模型 | 命令中的模型名 | 物理卡数 | 当前可做什么 |
| --- | --- | ---: | --- |
| BEVFormer | `bevformer` | 2 | 公开 nuScenes mini 训练 1 epoch |
| Sparse4D | `sparse4d` | 2 | 公开 nuScenes mini 训练 10 步 |
| Qwen3-VL-8B | `qwen3vl8b` | 2 | 演示图文数据 LoRA 训练 10 步 |
| WorldEngine | `worldengine` | 4 | 官方 NAVTEST 功能子集，AlgEngine 训练 10 步 |
| Apollo VisionNet | `apollo_visionnet` | 4 | 官方 OCC / nuScenes mini 功能子集，训练 10 步 |

## 2. 复制对应命令

### BEVFormer / Sparse4D：先检查，再训练

```bash
bash tools/quickstart.sh bevformer check
bash tools/quickstart.sh bevformer plan
bash tools/quickstart.sh bevformer mini
```

Sparse4D 将三条命令里的 `bevformer` 换成 `sparse4d`。`plan` 中的 `missing_data` 必须为空才能运行 `mini`；已有 mini 数据和权重时不会重新下载。

### Qwen3-VL-8B：先校验权重，再训练

```bash
bash tools/quickstart.sh qwen3vl8b check
bash tools/quickstart.sh qwen3vl8b download
bash tools/quickstart.sh qwen3vl8b plan
bash tools/quickstart.sh qwen3vl8b mini
```

`download` 使用 Qwen 官方 ModelScope 固定快照，支持续传，四个权重分片均校验 SHA256。默认目录是 `/var/ai-model/lzd/weights/qwen3-vl-8b-instruct`。`mini` 使用上游 6 条 `mllm_demo`，结束后自动检查 10 步 loss、LoRA 参数更新、优化器及两个 rank 的保存文件。

### WorldEngine：下载真实样本，再训练

```bash
bash tools/quickstart.sh worldengine check
bash tools/quickstart.sh worldengine download
bash tools/quickstart.sh worldengine plan
bash tools/quickstart.sh worldengine mini
```

`download` 获取约 **5.59 GB** 官方图像归档、NAVTEST 标注、PDM 缓存和感知基座权重，另取约 4 MB 轨迹词表；逐文件校验发布方 SHA256，准备 140 帧 / 64 个时序锚点。已有完整文件会重新校验后复用，中断后重跑同一命令续传。建议数据目录至少有 20 GB 空间。

`mini` 保留原模型结构、正式图像流水线和轨迹标签，加载官方感知基座，运行 FP32 四卡 10 步。NAVTEST 样本复用为功能训练，**不输出独立验证集精度**。准备好的 1# 可直接从 `plan` 开始。

### Apollo VisionNet：1# 已有真实 mini 功能数据

```bash
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
```

当前 1# 已准备 162 帧官方 VAL 标注及对应完整 OCC 标签，保留原始标签和 CAN bus。这些 VAL 样本仅复用做功能训练，**没有独立验证精度**；完整 OCC 归档尚未完成下载。`mini` 保留原配置三帧 / 六相机 / 图像缩放流程，随机初始化 DLA34，FP32 四卡训练 10 步。

其他机器先按 [Apollo 数据说明](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md)准备 nuScenes mini、官方 train/val pkl 及解压后的 OCC 目录，再执行：

```bash
bash tools/quickstart.sh apollo_visionnet prepare
bash tools/quickstart.sh apollo_visionnet plan
bash tools/quickstart.sh apollo_visionnet mini
```

仅有 VAL 标签时可显式使用 `APOLLO_VAL_ONLY=1 APOLLO_OCC_DIR=/解压后的/occ_gt_release_v1_0 bash tools/quickstart.sh apollo_visionnet prepare`，报告会标注 VAL 复用且不评估精度。1# 已准备好，无需再次执行 `prepare`。

WorldEngine / Apollo 的 `build` 只构建模型，`compute` 使用低分辨率合成输入检查计算，结果名为 `pass_compute_fixture`；真实样本短训练使用 `mini`。两种检查的范围见[实测记录](validation.md)。

## 3. 看结果

| 操作 | 看到什么才算这项检查通过 | 结果位置 |
| --- | --- | --- |
| `check` | `pass`，或仅管理面告警且已核对宿主的 `pass_with_warnings` | `outputs/<模型名>/environment.json` |
| `plan` | `missing_data: []` 才具备列出的文件 | 终端打印完整命令和缺失路径 |
| BEVFormer / Sparse4D `mini` | 正常退出、训练 loss 有限、checkpoint 可加载 | `outputs/bevformer/epoch_1.pth` / `outputs/sparse4d/iter_10.pth` |
| Qwen `mini` | 自动检查显示 `result: pass_functional` | `outputs/qwen3vl8b/checkpoint_check.json` |
| WorldEngine / Apollo `mini` | 正常退出；四个 rank 均为 `pass_training_smoke`，rank 0 的 `checkpoint_reload` 为 `pass`、优化器步数为 10 | `outputs/<模型名>/mini/training_check_rank{0,1,2,3}.json` 和 `iter_10.pth` |
| WorldEngine / Apollo `compute` | 四个 rank 均为 `pass_compute_fixture`；rank 0 重载通过 | `outputs/<模型名>/compute_fixture/rank{0,1,2,3}.json` |

已有实测结果与验证范围见[实测记录](validation.md)。日志里出现弃用提示不等于失败，最终要看退出码和上述检查结果；非有限输出 / loss / 梯度、参数未更新或保存检查失败会使 WorldEngine / Apollo 的 `mini` 返回失败。每次运行使用相同的输出目录和文件名，重新训练会覆盖该模型的 mini 输出；需要保留结果时先复制整个输出目录。

## 用的镜像

| 模型 | 1# 镜像 |
| --- | --- |
| BEVFormer、Sparse4D、Apollo | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| Qwen3-VL-8B | `local/llamafactory:a3-arm64-20261010` |
| WorldEngine | `local/migration-worldengine:26.1.0-a3-20261011` |

镜像 ID 会自动核对。官方下载地址、digest、具体 Python / CANN / TorchNPU 版本和构建步骤见[镜像与环境指南](environment.md)。新机器需先准备镜像和源码。

## 换卡、换目录和正式训练

默认选宿主逻辑设备：两卡 `0,2`，四卡 `0,2,4,6`。在 1# 上对应不同物理卡；脚本会拒绝重复选同一物理卡，并将容器内设备编号设为连续编号。先看宿主 `npu-smi info`，再修改，例如：

```bash
ASCEND_HOST_DEVICES=8,10,12,14 bash tools/quickstart.sh apollo_visionnet mini
```

改变代码 / 数据目录时，用 `MIGRATION_MOUNT_ROOT` 指定同时包含仓库和数据的父目录；该目录会按相同绝对路径挂载进容器。Qwen 另可设置 `QWEN_MODEL_DIR`，但须在这个挂载目录内。需要代理时，在当前 shell 设置现场批准的 `HTTPS_PROXY` / `HTTP_PROXY`，脚本只把它们传入下载容器。

正式训练使用 `train`，先按模型 README 准备完整数据、权重和独立验证集；`train` 会检查正式路径，`plan` 仅查看 mini 命令。Qwen 必须注册业务图文数据集并设置 `QWEN_DATASET_ID`。mini / 合成输入结果不能替代完整精度、长期稳定性或[官方硬件支持](environment.md)验收。

## 常见提示

| 提示 | 怎么处理 |
| --- | --- |
| `缺少镜像` / `镜像 ID 不匹配` | 按[环境指南](environment.md)导入已验证镜像；先核对来源，不修改预期 ID 来跳过检查。 |
| `missing_data` 有内容 | 按对应模型 README 准备列出的文件，再运行 `plan`。 |
| `NPU 设备编号...同一物理卡` | 查看宿主映射，选择不同物理卡的芯片。 |
| `pass_with_warnings` 且仅 `npu_smi_ok` 告警 | 查看宿主 `npu-smi info`，再结合实际模型计算结果；环境告警本身不代表训练通过。 |
| Qwen 下载中断 | 重跑同一条 `download` 续传；全部分片校验后才运行 `mini`。 |
| WorldEngine `mini` 缺少 prepared 文件 | 运行同一条 `download`；若官方文件已下载，也可运行 `worldengine prepare`。 |
| Apollo `prepare` 缺少 OCC 文件 | 按官方数据说明补齐标签；不要生成假标签。仅复用 VAL 时显式设置 `APOLLO_VAL_ONLY=1`。 |
| 提示卡不空闲 | 查看宿主 `npu-smi info`，换用空闲的不同物理卡；脚本不会停止其他服务。 |

模型说明：[BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md)、[Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md)、[Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md)、[WorldEngine](../PyTorch/contrib/cv/others/WorldEngine/README.md)、[Apollo](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md)。
