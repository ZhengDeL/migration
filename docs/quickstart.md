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
| WorldEngine | `worldengine` | 4 | AlgEngine 合成输入前后向与参数同步检查 |
| Apollo VisionNet | `apollo_visionnet` | 4 | BEV + OCC 合成输入前后向与参数更新检查 |

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

### WorldEngine：先准备轨迹词表，再检查四卡计算

```bash
bash tools/quickstart.sh worldengine check
bash tools/quickstart.sh worldengine download
bash tools/quickstart.sh worldengine compute
```

这里的 `download` **仅获取 4 MB 官方轨迹词表**并校验 SHA256。`compute` 使用合成四帧、八相机输入，检查 AlgEngine VADv2 计算；OpenScene 图像、训练标注、PDM 缓存和基座权重仍需另行准备。

### Apollo VisionNet：直接检查四卡计算

```bash
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet compute
```

`compute` 使用合成三帧、六相机输入，检查 DLA34 / BEV + OCC 计算。它使用随机初始化，不需要下载 nuScenes 或 OCC 数据。

WorldEngine / Apollo 也提供 `build`（只构建模型并放到 NPU）和 `plan`（列出真实训练缺少的文件）。这两个模型的 `compute` 已在 1# 四张物理卡上通过；结果名为 `pass_compute_fixture`，不代表真实数据训练或精度验收。

## 3. 看结果

| 操作 | 看到什么才算这项检查通过 | 结果位置 |
| --- | --- | --- |
| `check` | `pass`，或仅管理面告警且已核对宿主的 `pass_with_warnings` | `outputs/<模型名>/environment.json` |
| `plan` | `missing_data: []` 才具备列出的文件 | 终端打印完整命令和缺失路径 |
| BEVFormer / Sparse4D `mini` | 正常退出、训练 loss 有限、checkpoint 可加载 | `outputs/bevformer/epoch_1.pth` / `outputs/sparse4d/iter_10.pth` |
| Qwen `mini` | 自动检查显示 `result: pass_functional` | `outputs/qwen3vl8b/checkpoint_check.json` |
| WorldEngine / Apollo `compute` | 四个 rank 均为 `pass_compute_fixture`；rank 0 重载通过 | `outputs/<模型名>/compute_fixture/rank{0,1,2,3}.json` |

已有实测结果与验证范围见[实测记录](validation.md)。日志里出现弃用提示不等于失败，最终要看退出码和上述检查结果；非有限 loss、梯度或参数未更新会使 `compute` 返回失败。

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
ASCEND_HOST_DEVICES=8,10,12,14 bash tools/quickstart.sh apollo_visionnet compute
```

改变代码 / 数据目录时，用 `MIGRATION_MOUNT_ROOT` 指定同时包含仓库和数据的父目录；该目录会按相同绝对路径挂载进容器。Qwen 另可设置 `QWEN_MODEL_DIR`，但须在这个挂载目录内。需要代理时，在当前 shell 设置现场批准的 `HTTPS_PROXY` / `HTTP_PROXY`，脚本只把它们传入下载容器。

正式训练使用 `train`，先按模型 README 准备完整数据、权重和独立验证集，再核对 `plan`。Qwen 必须注册业务图文数据集并设置 `QWEN_DATASET_ID`。mini / 合成输入结果不能替代完整精度、长期稳定性或[官方硬件支持](environment.md)验收。

## 常见提示

| 提示 | 怎么处理 |
| --- | --- |
| `缺少镜像` / `镜像 ID 不匹配` | 按[环境指南](environment.md)导入已验证镜像；先核对来源，不修改预期 ID 来跳过检查。 |
| `missing_data` 有内容 | 按对应模型 README 准备列出的文件，再运行 `plan`。 |
| `NPU 设备编号...同一物理卡` | 查看宿主映射，选择不同物理卡的芯片。 |
| `pass_with_warnings` 且仅 `npu_smi_ok` 告警 | 查看宿主 `npu-smi info`，再结合实际模型计算结果；环境告警本身不代表训练通过。 |
| Qwen 下载中断 | 重跑同一条 `download` 续传；全部分片校验后才运行 `mini`。 |
| WorldEngine / Apollo 的 `mini` 提示没有真实 mini 数据 | 使用 `compute` 检查计算路径，真实数据另按模型说明准备。 |

模型说明：[BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md)、[Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md)、[Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md)、[WorldEngine](../PyTorch/contrib/cv/others/WorldEngine/README.md)、[Apollo](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md)。
