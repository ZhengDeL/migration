# 环境与镜像

本页整理日期：2026-10-19。下表来自九个模型自身的 `images.lock.json`；测试时间见[验证结果](validation.md)。其他机器优先导入对应完整 ZIP 的镜像归档，再按本模型 README 校验 image ID。

## 已验证的运行镜像

### DrivingSDK

适用模型：BEVFormer、Sparse4D、Apollo-Vision-Net、LSTM-Argoverse、STGCN-Traffic、ResWorld、MobileNetV2。

| 项目 | 固定值 |
| --- | --- |
| 镜像 tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| Docker image ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| cann | `9.1.0` |
| python | `3.10.21` |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post6` |
| mx_driving | `26.1.0.post2` |

镜像归档：`drivingsdk_headless.tar.gz`，9,272,909,176 字节，SHA256 `5ec0b1496efe124b5ad574a44822db84e7c3260e197e90f6c10ca4b4642a7f12`；归档中只有本模型所需的一个运行镜像。

官方 ARM64 来源 manifest：`swr.cn-south-1.myhuaweicloud.com/ascendhub/drivingsdk@sha256:52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801`。

[镜像锁与归档校验值](../PyTorch/contrib/cv/detection/BEVFormer/images.lock.json)。

### Qwen

适用模型：Qwen3-VL-8B。

| 项目 | 固定值 |
| --- | --- |
| 镜像 tag | `local/llamafactory:a3-arm64-20261012` |
| Docker image ID | `sha256:99d33965ae4502b07bffe2c3248887795ae0672a0e79dd0aaa567d09cce811b5` |
| cann | `9.1.0` |
| python | `3.12.13` |
| torch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post2` |
| transformers | `5.8.0` |
| peft | `0.18.1` |
| llamafactory | `0.9.6.dev0` |
| accelerate | `1.11.0` |

镜像归档：`qwen.tar.gz`，6,024,595,159 字节，SHA256 `0b0ce68cac12a215ad540ee9f458422cde3474a69810e5cd55fc2bb5f0c5faed`；归档中只有本模型所需的一个运行镜像。

官方 ARM64 来源 manifest：`quay.io/ascend/llamafactory@sha256:6394a17532563a9b978b2e524a20594287b8ee480fe159e4e57b5067ec8125ea`。

[镜像锁与归档校验值](../PyTorch/contrib/nlp/Qwen3-VL-8B/images.lock.json)。

### WorldEngine

适用模型：WorldEngine AlgEngine。

| 项目 | 固定值 |
| --- | --- |
| 镜像 tag | `local/migration-worldengine:26.1.0-a3-20261011` |
| Docker image ID | `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3` |
| cann | `9.1.0` |
| python | `3.10.21` |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post6` |
| mx_driving | `26.1.0.post2` |
| numpy | `1.23.5` |
| rasterio | `1.3.11` |
| fiona | `1.9.6` |

镜像归档：`worldengine.tar.gz`，9,935,906,014 字节，SHA256 `5cd585f49883906aac39d7dfdaf74ef9b458c3bf2fd4b3e5de00f4796aa91ecb`；归档中只有本模型所需的一个运行镜像。

[镜像锁与归档校验值](../PyTorch/contrib/cv/others/WorldEngine/images.lock.json)。

## 宿主环境与设备映射

实测宿主为 ARM64 Atlas 800I A3（`IT22HMDA_4_S`），驱动 26.1.1、固件 9.0.0.9.220。双芯片物理卡对应两个逻辑 NPU；本仓两卡/四卡试验选择不同物理卡的一个逻辑设备。用宿主 `npu-smi info` 核对物理映射和占用，再按模型 README 填写设备号。容器仅挂载所选设备时，容器内会重新连续编号，不能混用宿主和容器编号。

华为 [ModelZoo 环境安装指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md)区分 A3 训练与推理系列；本仓在推理系列上的实验不代表该产品取得官方训练支持。版本中的 `torch +cpu` 是配套基础 wheel 标记，实际 NPU 计算以实机记录为依据。

## 镜像构建来源

DrivingSDK 派生镜像使用 [headless Dockerfile](../docker/DrivingSDK-headless.Dockerfile)，固定 ARM64 OpenCV wheel，避免原无显示环境的 `libGL.so.1` 导入问题。WorldEngine 在它之上使用 [WorldEngine Dockerfile](../docker/WorldEngine.Dockerfile)、[固定额外依赖](../docker/worldengine_extra_requirements.txt)及 [ARM64 地理环境记录](../docker/worldengine_geo_linux-aarch64.explicit.txt)。Qwen 使用 [LLaMA-Factory 固定提交的 A3 容器方案](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md)。

Dockerfile 用于来源审计与重建，重新构建所得 image ID 可能不同，必须重新核对环境与验证。复现本仓既有结果使用模型锁定的归档和 image ID；滚动 `latest` 标签不能代表同一个已验证镜像。具体导入、环境检查、数据准备和训练命令均保留在各模型 README。
