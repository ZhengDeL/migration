# 910C/A3 环境与数据准备

## 先确认硬件和软件

在可信 SSH 主机指纹确认后，先只读采集 DMI 产品型号、`npu-smi info -t board -i 0`、`npu-smi info`、`/etc/os-release`、驱动/固件版本、CANN Toolkit/ops 版本、Python、PyTorch、`torch_npu`、`mx_driving`、容器及可用设备数。将具体产品型号与 [ModelZoo-PyTorch 训练硬件支持表](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md)核对：表中 Atlas A3 训练系列受支持，A3 推理系列标为不支持。`npu-smi info` 应同时核对 `NPU` 与 `Chip Phy-ID`：一张双芯片物理卡可对应两个逻辑设备，双卡任务要选择不同物理卡的逻辑 ID。以实际产品形态、SoC、操作系统和驱动配套选择镜像，不能仅凭任务表中的“910C”安装。服务器地址和登录信息仅在受控环境中保存，不写入公开仓库。

[DrivingSDK v26.1.0 安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)列出的 A3/openEuler 镜像是 `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10`。本项目 2026-10-10 在 1# 按 linux/arm64 manifest digest `sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801` 拉取并导入 Docker。容器内 `torch2.7.1`、`bevformer` 和 `sparse4d` 环境均为 Python 3.10.21、PyTorch 2.7.1、TorchNPU 2.7.1.post6、mx_driving 26.1.0.post2；两个模型环境分别预装了相应 OpenMMLab 依赖。宿主驱动、固件、设备挂载以及 NPU 算子仍须由实机检查确认，包版本检查不代表训练通过。

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

下载网络及代理由现场配置；Dockerfile 自身不访问网络，也不修改宿主软件。使用派生镜像前仍须做有设备的 `check_env.py`、算子和 HCCL 检查。

Qwen3-VL 使用独立的 [LLaMA-Factory A3 容器方案](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)。与本仓锁定的 LLaMA-Factory 源码同一提交的镜像说明列出 CANN 9.1.0、Python 3.12、PyTorch 2.10.0 和 TorchNPU 2.10.0.post2；`latest-a3-openeuler` 是会变化的标签，不能直接作为可复现版本。Qwen 镜像尚未在本项目机器上拉取或验证，选定后应固定 arm64 digest，再核对容器内实际包版本。DrivingSDK 的 PyTorch 2.7.1 环境与 Qwen 环境分别使用。

若不用官方预置环境，按各模型 README 和 DrivingSDK 补丁**源码编译** MMCV、mmdet、mmdet3d；不能直接安装未适配的 `mmcv-full` CUDA wheel。WorldEngine 与 Apollo 的旧版 OpenMMLab 依赖必须分别隔离并做 API/算子兼容验证，不能用 BEVFormer 环境导入成功作为它们训练可用的证据。

WorldEngine 官方 AlgEngine 文档指定 Python 3.9、CUDA PyTorch 2.0.1、MMCV 1.6.2、MMDetection3D 1.0.0rc6；Apollo 官方安装文档指定 Python 3.7、PyTorch 1.10、MMCV 1.4.1、MMDetection3D 0.18.1。它们不能直接照搬到当前 TorchNPU/DrivingSDK A3 镜像。`ascend.patch` 只覆盖应用层主要 NPU 入口，尚需建立可导入、可编译且算子支持的完整依赖矩阵。

## 数据与权重

公开源码和 Qwen3-VL-8B-Instruct 权重可在训练服务器直接下载：`tools/fetch_sources.py` 克隆 `sources.lock.json` 锁定的提交，`tools/fetch_qwen.py` 通过 `huggingface_hub.snapshot_download` 下载锁定的模型快照并核对权重分片。两个脚本默认只输出计划，执行时加 `--execute`。Qwen 下载脚本需在独立环境中预装 `huggingface_hub`，并预留模型权重和缓存空间。nuScenes、OpenScene 等数据集不由这些脚本自动下载；按上游许可和登录流程准备。

| 模型 | 训练前必须由数据负责人提供 |
| --- | --- |
| BEVFormer | nuScenes v1.0 full、CAN bus、`nuscenes_infos_temporal_{train,val}.pkl`、`r101_dcn_fcos3d_pretrain.pth`，参见 [官方示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md)。 |
| Sparse4D | nuScenes、`nuscenes_infos_{train,val}.pkl`、`nuscenes_kmeans900.npy`、ResNet50 预训练权重；旧版官方权重须按[模型说明](../PyTorch/contrib/cv/detection/Sparse4D/README.md)转换格式，参见 [DrivingSDK 示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md)。 |
| Qwen3-VL-8B | `Qwen3-VL-8B-Instruct` 本地完整权重/processor、按 LLaMA-Factory `data/dataset_info.json` 注册的图文训练集、独立验证集；`mllm_demo` 仅用于 smoke。 |
| WorldEngine | 官方 OpenScene/NAVSIM 数据、已处理的 AlgEngine 标注、`test_8192_kmeans.npy` 和配置引用的基座 checkpoint，参见 [数据指南](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/data_organization.md)。 |
| Apollo VisionNet | nuScenes、CAN bus、OCC 标注/生成的 train/val pkl、DLA34 预训练权重，参见 [官方数据指南](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/prepare_dataset.md)。 |

不要把数据、权重、训练输出或登录凭据提交到 GitHub。大数据通过挂载或符号链接放到上游配置所需的路径。`tools/run_train.py` 的 `missing_data` 输出可检查核心文件；还应人工确认文件完整性和许可证。

### 公开 mini 功能试验

[nuScenes 官方教程](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)提供无需完整 trainval 归档的 `v1.0-mini.tgz`。下载后校验传输完整性并解压到独立数据目录，再分别在 BEVFormer 和 Sparse4D 源码目录建立 `data/nuscenes` 链接。运行两仓上游转换器时显式选择 `v1.0-mini`；Sparse4D 还需由 mini 标注生成试验用 `nuscenes_kmeans900.npy`。训练入口用 `--smoke --dataset-version v1.0-mini`，其配置只用于功能验证。BEVFormer 缺少单独的 CAN bus 扩展时，上游转换器会把缺失场景的 CAN bus 信息填零，不能由该结果推断完整数据集精度。完整训练应重新准备 trainval、CAN bus、标注和锚框。

## 执行与留证顺序

1. 确认 SSH 主机指纹、设备拓扑、环境版本和可用显存；在独立容器/环境完成软件安装。
2. 校验 `sources.lock.json` 中提交并应用补丁；保存 `git diff --check` 与补丁 SHA256。
3. 先跑 `tools/check_env.py`，再用 `tools/run_train.py --smoke --execute` 检查各 rank 前向、反向、有限 loss 和参数更新。
4. 对变形注意力做前向/反向数值对齐，并对检测/占用/图文输出做代表性样本回归；逐模型启动完整训练与验证。
5. 记录吞吐量、显存、mAP/NDS/mIoU 或图文评估、checkpoint 和失败栈。若仅 smoke 通过，状态仍是“功能初验”。
