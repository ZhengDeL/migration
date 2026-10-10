# Ascend 910C 训练迁移样例

目录参考 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的 `PyTorch/contrib/{cv,nlp}/<类别>/<模型>` 组织方式。每个模型目录保留独立 README、迁移补丁或配置、`test/` 启动脚本；共享的版本锁定和预检工具位于仓库根目录。以下代码截至 2026-10-10 只完成本地静态验证，没有把任何模型标记为 910C 实机训练通过。

## 范围和状态

| 类别与模型 | 计划卡数 | 本仓范围 | 当前证据 |
| --- | ---: | --- | --- |
| [cv/detection/BEVFormer](PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2 | DrivingSDK 官方 NPU 补丁，FP16 训练 | 补丁逐文件校验和本地命令检查通过；待 NPU 实训 |
| [cv/detection/Sparse4D](PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2 | DrivingSDK 官方 NPU 补丁，2 卡 batch、迭代数和学习率 | 补丁逐文件校验、配置生成通过；待 NPU 实训 |
| [nlp/Qwen3-VL-8B](PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2 | LLaMA-Factory BF16 LoRA SFT | YAML 和命令静态检查通过；待权重、业务数据和 NPU 实训 |
| [cv/others/WorldEngine](PyTorch/contrib/cv/others/WorldEngine/README.md) | 4 | AlgEngine VADv2 50% 的 NPU 入口与变形注意力补丁 | 补丁可应用、代码可编译；SimEngine 和完整后训练链路未适配 |
| [cv/detection/Apollo-Vision-Net](PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 | 官方 BEV+OCC 训练仓的 DLA34 配置与变形注意力补丁 | 补丁可应用、代码可编译；待旧版 MMDetection3D 兼容性验证 |

研究依据和限制见 [社区调研](docs/community_research.md)，环境版本和数据准备见 [环境指南](docs/environment.md)。`sources.lock.json` 是唯一的上游版本清单；各模型目录存放本仓配置和补丁。上游源码、权重、数据、训练输出及服务器地址不纳入本仓。1# 计划使用 2 卡，2# 计划使用 4 卡，具体设备需在训练前确认。

## 本地准备

可以直接在训练服务器下载公开源码和 Qwen 权重。下载脚本默认只打印计划，追加 `--execute` 才联网；先在 1# 下载 DrivingSDK、BEVFormer、Sparse4D、LLaMA-Factory，后续在 2# 下载 WorldEngine 和 Apollo-Vision-Net。nuScenes、OpenScene 和预训练权重按各上游的数据获取流程准备。以下命令从仓库根目录执行：

```bash
python3 tools/fetch_sources.py drivingsdk bevformer sparse4d qwen3vl8b --execute
python3 -m pip install huggingface_hub
python3 tools/fetch_qwen.py --execute
```

脚本以 `sources.lock.json` 的完整提交或模型快照 SHA 下载，并存入被 Git 忽略的 `sources/`。如果训练机不能直连，可从可信机器传输相同快照。完成下载后应用补丁：

```bash
python3 tools/prepare.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --apply
python3 tools/prepare.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --apply
python3 tools/prepare.py qwen3vl8b --source sources/LLaMA-Factory
python3 tools/prepare.py worldengine --source sources/WorldEngine --apply
python3 tools/prepare.py apollo_visionnet --source sources/Apollo-Vision-Net --apply
```

脚本会校验 Git origin、完整提交 SHA、补丁的 LF 规范化 SHA256 和补丁后的逐文件内容；它也会统一 DrivingSDK 补丁的换行格式，对已应用补丁重复运行不会叠加。Sparse4D 另会复制并校验 DrivingSDK 的 `tools/patch.py` 和官方训练脚本。各模型必须使用隔离环境，特别是 Qwen 的 LLaMA-Factory 环境与两个旧版 OpenMMLab 栈。

## 训练前预检

登录目标容器后先做只读环境检查，例如：

```bash
python3 tools/check_env.py --model bevformer --nproc 2 --output outputs/bevformer/environment.json
python3 tools/check_env.py --model sparse4d --nproc 2 --output outputs/sparse4d/environment.json
python3 tools/check_env.py --model qwen3vl8b --nproc 2 --output outputs/qwen3vl8b/environment.json
```

`result=pass` 只代表包可导入、版本主次号相符和设备可见，不代表模型训练成功。数据和权重仍须按 [环境指南](docs/environment.md) 放到源码要求的位置。

对 BEVFormer、WorldEngine、Apollo 的变形注意力先在实际 NPU 环境运行 `python3 tools/check_msda.py --output outputs/<model>/msda.json`，留存前向和三类梯度的数值对齐结果。`run_train.py --execute` 也会自动执行这项检查，失败时不会启动训练。

## 训练命令

以下命令缺省只打印命令、缺失文件及计划；追加 `--execute` 才会启动。Qwen 的 `missing_data` 也会报告模型目录和数据集注册文件。输出、模型路径和数据集名使用实际目录，脚本不会自动下载权重或数据。

```bash
python3 tools/run_train.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --devices 0,1 --output-dir outputs/bevformer --smoke
python3 tools/run_train.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --devices 0,1 --output-dir outputs/sparse4d --smoke
python3 tools/run_train.py qwen3vl8b --source sources/LLaMA-Factory --devices 0,1 --output-dir outputs/qwen3vl8b --model-path /path/to/Qwen3-VL-8B-Instruct --dataset-id my_multimodal_dataset --smoke
```

先执行 smoke 并核对两个 rank 的前向、反向、有限 loss、参数更新与日志，再去掉 `--smoke` 进行完整训练。4 卡命令同理将 `--devices` 改为四个设备 ID，并将模型改为 `worldengine` 或 `apollo_visionnet`；它们必须完成算子和精度回归后才能进入完整训练。

## 验收边界

每个模型分别留存环境 JSON、固定提交/补丁 SHA、数据和权重清单、各 rank 训练日志、有限 loss、checkpoint，以及对应指标。BEVFormer 和 Sparse4D 需核对 nuScenes mAP/NDS；Apollo 需核对检测 mAP 和占用 mIoU；WorldEngine 需核对 AlgEngine 开环指标并另行验证 SimEngine 闭环链路；Qwen3-VL 需用独立图文样本评估回答内容。仅有进程存活或 smoke loss 不构成训练迁移验收。
