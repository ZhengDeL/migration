# 910C/A3 环境与数据准备

## 先确认硬件和软件

在可信 SSH 主机指纹确认后，先只读采集 DMI 产品型号、`npu-smi info -t board -i 0`、`npu-smi info`、`/etc/os-release`、驱动/固件版本、CANN Toolkit/ops 版本、Python、PyTorch、`torch_npu`、`mx_driving`、容器及可用设备数。将具体产品型号与 [ModelZoo-PyTorch 训练硬件支持表](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md)核对：表中 Atlas A3 训练系列受支持，A3 推理系列标为不支持。以实际产品形态、SoC、操作系统和驱动配套选择镜像，不能仅凭任务表中的“910C”安装。服务器地址和登录信息仅在受控环境中保存，不写入公开仓库。

[DrivingSDK v26.1.0 安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)列出的 A3/openEuler 候选镜像是 `swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk:26.1.0-9.1.0-torch_npu2.7.1.post6-a3-openeuler24.03-py3.10`。其中预置 `bevformer`、`sparse4d` 隔离环境；宿主驱动、固件和设备挂载仍须匹配。此标签只是候选，实际使用前须记录所选架构的镜像 digest、容器内版本和 NPU 自检结果。

Qwen3-VL 使用独立的 [LLaMA-Factory A3 容器方案](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)。与本仓锁定的 LLaMA-Factory 源码同一提交的镜像说明列出 CANN 9.1.0、Python 3.12、PyTorch 2.10.0 和 TorchNPU 2.10.0.post2；`latest-a3-openeuler` 是会变化的标签，不能直接作为可复现版本。Qwen 镜像尚未在本项目机器上拉取或验证，选定后应固定 arm64 digest，再核对容器内实际包版本。DrivingSDK 的 PyTorch 2.7.1 环境与 Qwen 环境分别使用。

若不用官方预置环境，按各模型 README 和 DrivingSDK 补丁**源码编译** MMCV、mmdet、mmdet3d；不能直接安装未适配的 `mmcv-full` CUDA wheel。WorldEngine 与 Apollo 的旧版 OpenMMLab 依赖必须分别隔离并做 API/算子兼容验证，不能用 BEVFormer 环境导入成功作为它们训练可用的证据。

WorldEngine 官方 AlgEngine 文档指定 Python 3.9、CUDA PyTorch 2.0.1、MMCV 1.6.2、MMDetection3D 1.0.0rc6；Apollo 官方安装文档指定 Python 3.7、PyTorch 1.10、MMCV 1.4.1、MMDetection3D 0.18.1。它们不能直接照搬到当前 TorchNPU/DrivingSDK A3 镜像。`ascend.patch` 只覆盖应用层主要 NPU 入口，尚需建立可导入、可编译且算子支持的完整依赖矩阵。

## 数据与权重

公开源码和 Qwen3-VL-8B-Instruct 权重可在训练服务器直接下载：`tools/fetch_sources.py` 克隆 `sources.lock.json` 锁定的提交，`tools/fetch_qwen.py` 通过 `huggingface_hub.snapshot_download` 下载锁定的模型快照并核对权重分片。两个脚本默认只输出计划，执行时加 `--execute`。Qwen 下载脚本需在独立环境中预装 `huggingface_hub`，并预留模型权重和缓存空间。nuScenes、OpenScene 等数据集不由这些脚本自动下载；按上游许可和登录流程准备。

| 模型 | 训练前必须由数据负责人提供 |
| --- | --- |
| BEVFormer | nuScenes v1.0 full、CAN bus、`nuscenes_infos_temporal_{train,val}.pkl`、`r101_dcn_fcos3d_pretrain.pth`，参见 [官方示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md)。 |
| Sparse4D | nuScenes、`nuscenes_infos_{train,val}.pkl`、`nuscenes_kmeans900.npy`、ResNet50 预训练权重，参见 [官方示例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md)。 |
| Qwen3-VL-8B | `Qwen3-VL-8B-Instruct` 本地完整权重/processor、按 LLaMA-Factory `data/dataset_info.json` 注册的图文训练集、独立验证集；`mllm_demo` 仅用于 smoke。 |
| WorldEngine | 官方 OpenScene/NAVSIM 数据、已处理的 AlgEngine 标注、`test_8192_kmeans.npy` 和配置引用的基座 checkpoint，参见 [数据指南](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/data_organization.md)。 |
| Apollo VisionNet | nuScenes、CAN bus、OCC 标注/生成的 train/val pkl、DLA34 预训练权重，参见 [官方数据指南](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/prepare_dataset.md)。 |

不要把数据、权重、训练输出或登录凭据提交到 GitHub。大数据通过挂载或符号链接放到上游配置所需的路径。`tools/run_train.py` 的 `missing_data` 输出可检查核心文件；还应人工确认文件完整性和许可证。

## 执行与留证顺序

1. 确认 SSH 主机指纹、设备拓扑、环境版本和可用显存；在独立容器/环境完成软件安装。
2. 校验 `sources.lock.json` 中提交并应用补丁；保存 `git diff --check` 与补丁 SHA256。
3. 先跑 `tools/check_env.py`，再用 `tools/run_train.py --smoke --execute` 检查各 rank 前向、反向、有限 loss 和参数更新。
4. 对变形注意力做前向/反向数值对齐，并对检测/占用/图文输出做代表性样本回归；逐模型启动完整训练与验证。
5. 记录吞吐量、显存、mAP/NDS/mIoU 或图文评估、checkpoint 和失败栈。若仅 smoke 通过，状态仍是“功能初验”。
