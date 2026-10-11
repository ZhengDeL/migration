# 910C/A3 环境与数据准备

## 先找到要用的镜像

1# 已准备下表镜像，直接按[快速上手](quickstart.md)运行。`local/...` 是本地标签，其他服务器需要导入已验证的镜像归档或按下文构建。

| 模型 | 实际镜像 | 实测版本 |
| --- | --- | --- |
| BEVFormer、Sparse4D、Apollo | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` | CANN 9.1.0 / Python 3.10.21 / Torch 2.7.1 / TorchNPU 2.7.1.post6 / DrivingSDK 26.1.0.post2 |
| Qwen3-VL-8B | `local/llamafactory:a3-arm64-20261011` | CANN 9.1.0 / Python 3.12.13 / Torch 2.10.0 / TorchNPU 2.10.0.post2 / transformers 5.8.0 |
| WorldEngine | `local/migration-worldengine:26.1.0-a3-20261011` | 上述 DrivingSDK 栈，加隔离的 AlgEngine / nuPlan 地理依赖 |

`quickstart.sh` 核对的本地 Docker image ID：

```text
DrivingSDK headless: sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2
Qwen:               sha256:5134fb97fcfe4a9a83103fbe45c6f196b847e4cfac2f873a29fb4effab8e7645
WorldEngine:        sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3
```

上游 registry digest 标识拉取的 ARM64 manifest，本地 image ID 标识导入或构建后的镜像；两者不能互换。完整实测范围见[验证记录](validation.md)。

## 先确认硬件和软件

在可信 SSH 主机指纹确认后，先只读采集 DMI 产品型号、`npu-smi info -t board -i 0`、`npu-smi info`、`/etc/os-release`、驱动/固件版本、CANN Toolkit/ops 版本、Python、PyTorch、`torch_npu`、`mx_driving`、容器及可用设备数。将具体产品型号与 [ModelZoo-PyTorch 训练硬件支持表](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md)核对：表中 Atlas A3 训练系列受支持，A3 推理系列标为不支持。`npu-smi info` 应同时核对 `NPU` 与 `Chip Phy-ID`：一张双芯片物理卡可对应两个逻辑设备，双卡任务要选择不同物理卡的逻辑 ID。以实际产品形态、SoC、操作系统和驱动配套选择镜像，不能仅凭任务表中的“910C”安装。服务器地址和登录信息仅在受控环境中保存，不写入公开仓库。

1# 实测 DMI 型号为 Atlas 800I A3，属于推理系列；本仓的双卡 mini 成功是这台机器上的实验性功能证据，不改变上述官方训练支持边界。正式训练硬件验收仍需受支持的训练系列设备或厂商对该型号的明确支持依据。

Docker 只挂载所选设备时，再核对**容器内**的编号。1# 实测将宿主 `/dev/davinci0`、`/dev/davinci2` 挂载进容器后，TorchNPU 看到两个连续设备 `0,1`；容器内设 `ASCEND_RT_VISIBLE_DEVICES=0,1` 时两者均可见，若设为宿主编号 `0,2` 则只可见一个。`run_train.py --devices` 和各启动脚本的 `ASCEND_DEVICES` 填进程/容器内编号。宿主设备映射及当前资源占用仍以宿主 `npu-smi info` 判断。

[DrivingSDK v26.1.0 安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)列出的 A3/openEuler 镜像是 `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10`。本项目 2026-10-10 在 1# 按 linux/arm64 manifest digest `sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` 拉取并导入 Docker，原镜像 ID 为 `sha256:1a63c030ea4036467cfc486bef006acf6f1cfbb1cc3b0dc362d07f86b84349bd`。容器内 `torch2.7.1`、`bevformer` 和 `sparse4d` 环境均为 Python 3.10.21、PyTorch 2.7.1+cpu、TorchNPU 2.7.1.post6、mx_driving 26.1.0.post2；`+cpu` 是配套 PyTorch 包版本标记，实际 NPU 计算须单独验证。两个模型环境分别预装了相应 OpenMMLab 依赖。1# 宿主驱动 26.1.1、固件 9.0.0.9.220；实机双 rank 前后向、参数更新与 HCCL，以及变形注意力前后向数值检查已通过。

该官方镜像的两个模型环境同时装有 `opencv-python` 和 `opencv-python-headless` 4.11.0.86；无设备容器内导入 MMCV 时，当前 `cv2` 二进制缺少 `libGL.so.1`。本仓的 [派生镜像 Dockerfile](../docker/DrivingSDK-headless.Dockerfile)只在 `bevformer`、`sparse4d` 环境中保留 SHA256 固定的 ARM64 headless wheel，保留原镜像供回退。先按 Dockerfile 指定的版本下载 wheel 到被 Git 忽略的 `wheelhouse/`，校验 `sha256:6efabcaa9df731f29e5ea9051776715b1bdd1845d7c9530065c7951d2a2899eb`，然后在已导入原镜像的机器上离线构建：

```bash
mkdir -p wheelhouse
docker run --rm -v "$PWD/wheelhouse":/wheelhouse:rw \
  -e HTTPS_PROXY -e HTTP_PROXY \
  --entrypoint /opt/conda/envs/bevformer/bin/python \
  local/drivingsdk:26.1.0-a3-arm64-20261010 \
  -m pip download --no-deps --only-binary=:all: --dest /wheelhouse \
  opencv-python-headless==4.11.0.86
docker build --network none -f docker/DrivingSDK-headless.Dockerfile \
  -t local/migration-drivingsdk:26.1.0-a3-headless-20261010 .
```

下载网络及代理由现场配置；Dockerfile 自身不访问网络，也不修改宿主软件。1# 实际构建的派生镜像标签为 `local/migration-drivingsdk:26.1.0-a3-headless-20261010`，镜像 ID 为 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。BEVFormer 与 Sparse4D 使用该镜像完成了双卡 mini 功能训练。两套容器的 `check_env.py` 均报告 `pass_with_warnings`：包、NPU 数量和算子导入通过；容器内 `npu-smi info` 返回 211/`-9005`，宿主机 `npu-smi info` 正常。容器管理面告警与实际计算结果须分别留证。

Qwen3-VL 使用独立的 [LLaMA-Factory A3 容器方案](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)。2026-10-11 重新核验并在 1# 将 `quay.io/ascend/llamafactory:latest-a3-openeuler` 的 linux/arm64 manifest 固定为 `quay.io/ascend/llamafactory@sha256:f1aa74bf63e62cd9503f8b34922b6a26f14c49bf1910b484b44cd57d624cfad1`；导入 Docker 后镜像 ID 为 `sha256:5134fb97fcfe4a9a83103fbe45c6f196b847e4cfac2f873a29fb4effab8e7645`，本地标签 `local/llamafactory:a3-arm64-20261011`。容器内实测 CANN 9.1.0、Python 3.12.13、PyTorch 2.10.0+cpu、TorchNPU 2.10.0.post2、transformers 5.8.0、LLaMA-Factory 0.9.6.dev0。Docker inspect 的镜像安装大小为 15,072,572,748 字节（不是传输归档大小）；单独导出文件的 SHA256 为 `09238b0643c89c259a95058c0b58cebd9dede37f93463ffd33ee2a32822aa57a`。最终交付采用包含五个镜像标签、共享层去重的压缩归档；分卷与 SHA256 以交付 `manifest.json` 为准。2026-10-11 新镜像已重新完成 Qwen3-VL-8B 双卡 LoRA 十步训练、checkpoint/参数更新检查和保存后的单图推理；基础双 rank NPU/HCCL 检查在此前镜像完成。业务数据训练与精度尚未验收。旧 ARM64 manifest 本次查询返回 404，旧镜像仅作为归档回退版本；标签会变化，后续复用以上 digest 或已校验归档。`run_train.py` 将已锁定 LLaMA-Factory 源码的 `src` 放在 `PYTHONPATH` 首位。该镜像预置空的 `http_proxy`/`https_proxy`；若现场经批准使用命令级代理下载权重，需在该容器中同时设置小写和大写变量，并以实际请求确认生效。Qwen [官方 ModelScope 仓](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)已提供相同模型，本仓另外锁定其提交 `5d854aab08710c16b980ec6d603d863b3821b915`；从该源下载后必须逐一核对四个分片与 Hugging Face 锁定快照的 SHA256，不能仅凭模型名判断内容相同。DrivingSDK 的 PyTorch 2.7.1 环境与 Qwen 环境分别使用。

### Apollo 的实际环境

Apollo 使用 DrivingSDK headless 镜像的 `bevformer` 环境，MMCV 1.7.2 / mmdet 2.24.0 / mmdet3d 1.0.0rc4 / mmseg 0.30.0。已通过模型构建、合成计算和官方 train/val 与 nuScenes mini 交集的四卡 10 步训练（242 train / 162 val），并重载检查全部 162 个独立验证帧。兼容补丁处理 NumPy 角度、文件句柄 PKL 加载和编译 FocalLoss 崩溃；NPU 使用 MMDetection 已有的 PyTorch 公式，并完成有限案例的 CPU/NPU 前后向对比。完整 OCC 归档已校验并安全解压；官方仓内 DLA 深度预训练权重经 Git blob、大小、SHA256 验证，训练前全部 222 个权重张量与模型完全匹配。完整 nuScenes 数据精度仍待验收。

### WorldEngine 的独立派生镜像

[WorldEngine.Dockerfile](../docker/WorldEngine.Dockerfile) 基于上述 DrivingSDK headless 镜像。额外 Python 包安装在 `/opt/worldengine-deps`，地理二进制依赖安装在 `/opt/worldengine-geo`，保留原 CANN 的 Python 搜索路径。NumPy 为 1.23.5，rasterio 为 1.3.11，Fiona 为 1.9.6；实测完整 conda 包清单见[ARM64 环境记录](../docker/worldengine_geo_linux-aarch64.explicit.txt)。

[额外依赖清单](../docker/worldengine_extra_requirements.txt)固定 62 项直接及传递运行依赖，用 `--no-deps` 安装。nuPlan devkit 来自 [Motional 官方仓](https://github.com/motional/nuplan-devkit/tree/ce3c323af01c0d7ec5672f7832ef53f9c679aab0)的固定提交 `ce3c323af01c0d7ec5672f7832ef53f9c679aab0`。旧 Lightning 使用配套 torchmetrics 0.2.0；motmetrics 1.4.0 适配 Python 3.10。Matplotlib 风格名通过可审计的模型兼容补丁处理。

在 ARM64 机器上、已准备 DrivingSDK headless 基础镜像后，从本仓根目录构建：

```bash
docker build --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  --build-arg http_proxy="${HTTP_PROXY:-}" --build-arg https_proxy="${HTTPS_PROXY:-}" \
  -f docker/WorldEngine.Dockerfile \
  -t local/migration-worldengine:26.1.0-a3-20261011 .
```

代理按现场网络配置；直连环境可省略代理参数。1# 最终镜像 ID 为上表 `ff85...`，在该镜像上已通过环境导入、58,207,822 参数模型构建、四卡合成计算，以及官方 NAVTEST 真实图像 / 轨迹 / PDM 缓存的四卡 10 步训练。真实短训练保留原图像流水线，加载官方感知基座；参数同步及模型 / 优化器保存状态检查通过。完整数据指标仍未验收。

WorldEngine 官方 CUDA 依赖和 Apollo 官方旧版 CUDA 依赖作为适配基线，本次 Ascend 实验使用以上已实测栈。若重新搭建 OpenMMLab，按 DrivingSDK 补丁**源码编译**；新构建可能产生不同 image ID，需先核对依赖并重新执行功能验证，再通过代码评审更新镜像锁定。直接导入已验证的 Docker 归档可保留原 image ID。

### 新机器的镜像准备

先核对 ARM64 架构、具体 A3 产品和驱动配套。以下为**准备命令示例**；1# 已安装镜像，无需重复拉取。

```bash
docker pull --platform linux/arm64 \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801
docker tag \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801 \
  local/drivingsdk:26.1.0-a3-arm64-20261010

docker pull --platform linux/arm64 \
  quay.io/ascend/llamafactory@sha256:f1aa74bf63e62cd9503f8b34922b6a26f14c49bf1910b484b44cd57d624cfad1
docker tag \
  quay.io/ascend/llamafactory@sha256:f1aa74bf63e62cd9503f8b34922b6a26f14c49bf1910b484b44cd57d624cfad1 \
  local/llamafactory:a3-arm64-20261011
```

Docker 拉取走宿主已有的 registry 网络配置；仅在 shell 设置代理不一定影响 Docker daemon。无法拉取时，使用可信机器上的 `docker save` 归档，经文件 SHA256 校验后在目标机 `docker load`。本仓不修改宿主 Docker daemon 或 DNS。DrivingSDK 继续按前文构建 headless 派生镜像，WorldEngine 再按独立 Dockerfile 构建。

镜像当前选择和实测 ID 统一由 [images.lock.json](../images.lock.json) 管理；版本核验日期、最新分支与已采用发布版的关系见[版本核验](latest_versions.md)。优先通过[离线恢复](offline_delivery.md)导入整套已验证镜像。

## 数据与权重

公开源码和 Qwen3-VL-8B-Instruct 权重可在训练服务器直接下载：`tools/fetch_sources.py` 克隆 `sources.lock.json` 锁定的提交，`tools/fetch_qwen.py` 从发布方 Hugging Face 或 ModelScope 仓下载固定快照，并逐一核对四个权重分片的 SHA256。两个脚本默认只输出计划，执行时加 `--execute`。在已准备好的 1# 上优先使用[快速上手](quickstart.md)中的 `bash tools/quickstart.sh qwen3vl8b download`，无需在宿主机安装 Python 包；该入口使用 Qwen 官方 ModelScope 仓。权重分片约 18 GB，下载临时文件还需要额外空间。独立执行 `fetch_qwen.py` 时需在其 Python 环境安装相应的 `huggingface_hub` 或 `modelscope`；可用 `--destination` 把模型放在仓库外的数据卷。WorldEngine 功能子集由 `worldengine download` 获取固定官方文件并校验；正式完整数据集按上游许可与下载流程准备；交付中已包含本次验证使用的完整 Apollo OCC 归档和公开 nuScenes mini。

| 模型 | 训练前必须由数据负责人提供 |
| --- | --- |
| BEVFormer | nuScenes v1.0 full、CAN bus、`nuscenes_infos_temporal_{train,val}.pkl`、`r101_dcn_fcos3d_pretrain.pth`，参见 [官方示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md)。 |
| Sparse4D | nuScenes、`nuscenes_infos_{train,val}.pkl`、`nuscenes_kmeans900.npy`、ResNet50 预训练权重；旧版官方权重须按[模型说明](../PyTorch/contrib/cv/detection/Sparse4D/README.md)转换格式，参见 [DrivingSDK 示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md)。 |
| Qwen3-VL-8B | `Qwen3-VL-8B-Instruct` 本地完整权重/processor、按 LLaMA-Factory `data/dataset_info.json` 注册的图文训练集、独立验证集；`mllm_demo` 仅用于 smoke。 |
| WorldEngine | 正式训练需官方完整 OpenScene/NAVSIM 图像、AlgEngine navtrain/navtest 标注、`pdm_8192_gt_cache_nav{train,test}.pkl`、词表和基座 checkpoint，见[固定上游数据指南](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/data_organization.md)。功能子集用 `worldengine download` 自动获取约 5.59 GB 固定文件及词表。 |
| Apollo VisionNet | nuScenes、CAN bus、完整 OCC 标注 / train/val pkl、DLA34 预训练权重，见[固定官方数据指南](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/prepare_dataset.md)。1# 已有完整 OCC 归档，mini 交集为 242 train / 162 val；DLA 权重已包含在固定官方源码的 `ckpts/` 内并核验。正式训练仍需完整 nuScenes 和 CAN bus。 |

不要把数据、权重、训练输出或登录凭据提交到 GitHub。大数据通过挂载或符号链接放到上游配置所需的路径。`tools/run_train.py` 的 `missing_data` 输出可检查核心文件；还应人工确认文件完整性和许可证。

### 公开 mini 功能试验

[nuScenes 官方教程](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)提供无需完整 trainval 归档的 `v1.0-mini.tgz`。下载后校验传输完整性并解压到独立数据目录，再分别在 BEVFormer 和 Sparse4D 源码目录建立 `data/nuscenes` 链接。BEVFormer 在无 NPU 的数据准备容器中使用[仓内纯数据入口](../PyTorch/contrib/cv/detection/BEVFormer/README.md)执行固定上游元数据函数；Sparse4D 直接运行其上游 `tools/nuscenes_converter.py --version v1.0-mini`，再由 mini 标注生成试验用 `nuscenes_kmeans900.npy`。训练入口用 `--smoke --dataset-version v1.0-mini`，其配置只用于功能验证。BEVFormer 缺少单独的 CAN bus 扩展时，上游转换函数会把缺失场景的 CAN bus 信息填零，不能由该结果推断完整数据集精度。完整训练应重新准备 trainval、CAN bus、标注和锚框。

## 执行与留证顺序

1. 确认 SSH 主机指纹、设备拓扑、环境版本和可用显存；在独立容器/环境完成软件安装。
2. 校验 `sources.lock.json` 中提交并应用补丁；保存 `git diff --check` 与补丁 SHA256。
3. 先跑 `tools/check_env.py`，再用 `tools/run_train.py --smoke --execute` 检查各 rank 前向、反向、有限 loss 和参数更新。
4. 对变形注意力做前向/反向数值对齐，并对检测/占用/图文输出做代表性样本回归；逐模型启动完整训练与验证。
5. 记录吞吐量、显存、mAP/NDS/mIoU 或图文评估、checkpoint 和失败栈。若仅 smoke 通过，状态仍是“功能初验”。
