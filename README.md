# Ascend 910C 训练迁移样例

目录参考 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的 `PyTorch/contrib/{cv,nlp}/<类别>/<模型>` 组织方式。每个模型目录保留独立 README、迁移补丁或配置、`test/` 启动脚本；共享的版本锁定和预检工具位于仓库根目录。截至 2026-10-10，本仓已在 1# 导入固定 digest 的 DrivingSDK 镜像并构建 headless 派生镜像，完成基础双 rank 与四 rank NPU/HCCL、变形注意力数值检查，以及 BEVFormer、Sparse4D 使用公开 nuScenes mini 的双卡模型训练功能试验。四 rank 是 1# 的基础通信与参数更新试验，不是 WorldEngine/Apollo 的模型级训练。上述结果不包含完整数据集精度或正式训练硬件支持验收。

**第一次使用：**先看[1# 双卡快速上手](docs/quickstart.md)。在已准备好镜像和数据的 1# 上，从仓库根目录执行 `bash tools/quickstart.sh bevformer check` 检查环境，再执行 `bash tools/quickstart.sh bevformer plan` 查看缺失文件；准备齐全后执行 `bash tools/quickstart.sh bevformer mini`。Sparse4D 将模型名换为 `sparse4d`；Qwen 先执行 `bash tools/quickstart.sh qwen3vl8b download` 下载并校验权重。

## 范围和状态

| 类别与模型 | 计划卡数 | 本仓范围 | 当前证据 |
| --- | ---: | --- | --- |
| [cv/detection/BEVFormer](PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2 | DrivingSDK 官方 NPU 补丁与 worker 序列化兼容补丁，FP16 训练 | 1# mini 162 迭代/1 epoch 正常退出，有限 loss、checkpoint 已核验；待完整数据精度 |
| [cv/detection/Sparse4D](PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2 | DrivingSDK 官方 NPU 补丁、普通 AdamW/逐参数裁剪兼容覆盖、2 卡配置 | 1# mini 10 迭代正常退出，有限 loss、checkpoint 已核验；待完整数据精度和性能 |
| [nlp/Qwen3-VL-8B](PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2 | LLaMA-Factory BF16 LoRA SFT | 上游源码与独立 A3 ARM64 镜像已在 1# 核对；该镜像双 rank 基础 NPU/HCCL、锁定源码的 YAML 参数解析通过，待权重、业务数据和模型级实训 |
| [cv/others/WorldEngine](PyTorch/contrib/cv/others/WorldEngine/README.md) | 4 | AlgEngine VADv2 50% 的 NPU 入口与变形注意力补丁 | 补丁可应用、代码可编译；SimEngine 和完整后训练链路未适配 |
| [cv/detection/Apollo-Vision-Net](PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 | 官方 BEV+OCC 训练仓的 DLA34 配置与变形注意力补丁 | 补丁可应用、代码可编译；待旧版 MMDetection3D 兼容性验证 |

研究依据和限制见 [社区调研](docs/community_research.md)，环境版本和数据准备见 [环境指南](docs/environment.md)。`sources.lock.json` 是唯一的上游版本清单；各模型目录存放本仓配置和补丁。上游源码、权重、数据、训练输出及服务器地址不纳入本仓。1# 已使用两张物理卡进行功能试验；2# 计划使用 4 卡，设备仍待确认。

## 实际镜像

| 模型 | 上游 ARM64 镜像（固定 digest） | 1# 实际运行镜像 | 验证范围 |
| --- | --- | --- | --- |
| BEVFormer、Sparse4D | `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` | `local/migration-drivingsdk:26.1.0-a3-headless-20261010`；ID `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` | 各自隔离的预置环境，双物理卡 mini 训练通过 |
| Qwen3-VL-8B | `quay.io/ascend/llamafactory@sha256:a95eeae05d63e2ec046c946a35301326aec742d914aed4bbe05caa572d478d36` | `local/llamafactory:a3-arm64-20261010`；ID `sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8` | 双物理卡基础 NPU/HCCL 与训练配置解析通过；模型级训练待权重 |
| WorldEngine、Apollo VisionNet | 尚未选定或构建 | 无 | 旧版 OpenMMLab 依赖未完成 A3 兼容验证，2# 未接入 |

DrivingSDK 官方标签、派生镜像 Dockerfile、包版本和构建方法，以及 Qwen 镜像版本清单见[环境指南](docs/environment.md)。`sha256:523...` 和 `sha256:a95...` 是上游 ARM64 manifest digest；`sha256:34a...` 和 `sha256:8aa...` 是 1# 本地 Docker image ID，两类标识不可互换。WorldEngine 与 Apollo 的补丁静态通过，不表示现有镜像可运行其训练。

## 本地准备

可以直接在训练服务器下载公开源码和 Qwen 权重。在已准备好的 1# 上，最省步骤的是[快速上手](docs/quickstart.md)的 `check → plan → mini`；Qwen 权重先运行 `bash tools/quickstart.sh qwen3vl8b download`，由固定镜像内的 Python 下载并校验。下面是新机器或需要重新准备源码时的详细步骤。仓库工具要求 Python 3.10+，应在相应的训练容器或独立环境中运行；宿主机系统 Python 版本不一定满足要求。下载脚本默认只打印计划，追加 `--execute` 才联网；先在 1# 下载 DrivingSDK、BEVFormer、Sparse4D、LLaMA-Factory，后续在 2# 下载 WorldEngine 和 Apollo-Vision-Net。若服务器需要代理，仅为下载命令配置经批准的 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量，不改系统 DNS 或 Docker 守护进程配置。nuScenes、OpenScene 和预训练权重按各上游的数据获取流程准备。以下命令从仓库根目录执行：

```bash
python3 tools/fetch_sources.py drivingsdk bevformer sparse4d qwen3vl8b --execute
bash tools/quickstart.sh qwen3vl8b download  # 在已安装固定 Qwen 镜像的 1# 上运行
```

脚本以 `sources.lock.json` 的完整提交或模型快照 SHA 下载，Qwen 四个权重分片另按固定 SHA256 校验。`quickstart.sh` 默认把 Qwen 权重放在仓库外的 `weights/`；独立运行 `fetch_qwen.py` 时可用 `--destination` 指定目录，且需先在对应 Python 环境安装 `modelscope` 或 `huggingface_hub`。如果训练机不能直连，可从可信机器传输相同快照。完成下载后应用补丁：

```bash
python3 tools/prepare.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --apply
python3 tools/prepare.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --apply
python3 tools/prepare.py qwen3vl8b --source sources/LLaMA-Factory
python3 tools/prepare.py worldengine --source sources/WorldEngine --apply
python3 tools/prepare.py apollo_visionnet --source sources/Apollo-Vision-Net --apply
```

脚本会校验 Git origin、完整提交 SHA、补丁的 LF 规范化 SHA256 和补丁后的逐文件内容；它也会统一 DrivingSDK 补丁的换行格式，对已应用补丁重复运行不会叠加。BEVFormer 在官方补丁后应用本仓可审计的 worker 序列化补丁；Sparse4D 在官方 `tools/patch.py` 的固定字节基础上生成普通 AdamW 与逐参数裁剪覆盖并核验生成结果。各模型必须使用隔离环境，特别是 Qwen 的 LLaMA-Factory 环境与两个旧版 OpenMMLab 栈。

## 训练前预检

启动目标容器前先根据宿主机 `npu-smi info` 的 `NPU`（物理卡）和 `Chip Phy-ID`（逻辑芯片）映射选择设备。两卡任务应选分属两张物理卡的芯片；例如宿主挂载 `/dev/davinci0` 和 `/dev/davinci2`，容器内实测为连续的逻辑索引 `0,1`，因此在**这个容器内**设置 `ASCEND_DEVICES=0,1`。若直接在宿主环境运行，则按宿主可见编号设置，不能把两种编号混用。还需把数据目录挂载到容器内相同的绝对路径，以保持源码目录的 nuScenes 符号链接有效。各 `test/train_*.sh` 要求显式设置此变量。随后做环境检查，例如：

```bash
export ASCEND_DEVICES=0,1  # 示例：容器只挂载宿主 davinci0 和 davinci2
export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
python3 tools/check_env.py --model bevformer --nproc 2 --output outputs/bevformer/environment.json
python3 tools/check_env.py --model sparse4d --nproc 2 --output outputs/sparse4d/environment.json
python3 tools/check_env.py --model qwen3vl8b --nproc 2 --output outputs/qwen3vl8b/environment.json
```

`result=pass` 只代表包可导入、版本主次号相符和设备可见，不代表模型训练成功。隔离容器中的 `npu-smi` 可能因管理面挂载不完整而失败；此时检查报告为 `pass_with_warnings`，须另行保存宿主 `npu-smi info`，并核对真实 NPU 计算与 HCCL。数据和权重仍须按 [环境指南](docs/environment.md) 放到源码要求的位置。

在目标镜像中先执行以下两 rank 前向、反向、参数更新和 HCCL 集合通信检查，并留存 `rank0.json`、`rank1.json`：

```bash
torchrun --standalone --nproc_per_node=2 tools/check_npu_training.py --output-dir outputs/base_npu
```

同一脚本也可在四张不同物理卡可见时用 `--nproc_per_node=4` 检查四 rank；1# 实测四 rank 的有限 loss、参数更新及 HCCL all-reduce sum=10 均通过。这个检查只证明基础算子和通信在当时可用；具体产品的官方训练支持边界仍按 [环境指南](docs/environment.md)判断，模型级训练和精度也须分别验证。

对 BEVFormer、WorldEngine、Apollo 的变形注意力先在实际 NPU 环境运行 `python3 tools/check_msda.py --output outputs/<model>/msda.json`，留存前向和三类梯度的数值对齐结果。`run_train.py --execute` 也会自动执行这项检查，失败时不会启动训练。

## 训练命令

以下命令缺省只打印命令、缺失文件及计划；追加 `--execute` 才会启动。Qwen 的 `missing_data` 也会报告模型目录和数据集注册文件。输出、模型路径和数据集名使用实际目录，脚本不会自动下载权重或数据。

```bash
python3 tools/run_train.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --devices "$ASCEND_DEVICES" --output-dir outputs/bevformer --smoke
python3 tools/run_train.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --devices "$ASCEND_DEVICES" --output-dir outputs/sparse4d --smoke
python3 tools/run_train.py qwen3vl8b --source sources/LLaMA-Factory --devices "$ASCEND_DEVICES" --output-dir outputs/qwen3vl8b --model-path /path/to/Qwen3-VL-8B-Instruct --dataset-id my_multimodal_dataset --smoke
```

先执行 smoke 并核对两个 rank 的前向、反向、有限 loss、参数更新与日志，再去掉 `--smoke` 进行完整训练。4 卡命令同理将 `--devices` 改为四个设备 ID，并将模型改为 `worldengine` 或 `apollo_visionnet`；它们必须完成算子和精度回归后才能进入完整训练。

BEVFormer 和 Sparse4D 可在准备好公开 nuScenes `v1.0-mini` 及各自标注后追加 `--smoke --dataset-version v1.0-mini` 做功能试验。该模式只运行有限训练，不执行完整数据集精度验收；转换命令与权重要求见各模型 README。

## 验收边界

每个模型分别留存环境 JSON、固定提交/补丁 SHA、数据和权重清单、各 rank 训练日志、有限 loss、checkpoint，以及对应指标。BEVFormer 和 Sparse4D 需核对 nuScenes mAP/NDS；Apollo 需核对检测 mAP 和占用 mIoU；WorldEngine 需核对 AlgEngine 开环指标并另行验证 SimEngine 闭环链路；Qwen3-VL 需用独立图文样本评估回答内容。仅有进程存活或 smoke loss 不构成训练迁移验收。
