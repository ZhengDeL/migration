# 版本核验（2026-10-11）

本交付冻结当日核验的上游提交、配套镜像与权重，恢复后得到相同文件。移动的 `main` / `latest` 不用于验收依据；重新更新依赖时需要重新验证模型。

## 源码与权重

| 项目 | 采用提交 | 当日核验结论 |
| --- | --- | --- |
| BEVFormer | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | 与官方默认分支 HEAD 相同 |
| Sparse4D | `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | 与官方 HEAD 相同；相比 SDK 基线仅 README 新增两行，运行代码相同，补丁在新提交上核验通过 |
| LLaMA-Factory | `1fded918ac4723ac9966363f51ed4500866c5d1d` | 与官方默认分支 HEAD 相同 |
| WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` | 与官方默认分支 HEAD 相同 |
| Apollo VisionNet | `4909aa681bc435a74e793e714b8066d9bbf0095d` | 与官方默认分支 HEAD 相同；DLA 权重就在该提交中，已核验官方 Git blob |
| Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 发布方 Hugging Face `main` 快照相同，四权重分片 SHA256 锁定 |
| WorldEngine 数据 | `d0e88d858533ca407a3387a26577c083aa7b3617` | 发布方 Hugging Face `main` 快照相同，四功能验证文件及词表 SHA256 锁定 |

完整来源见 [sources.lock.json](../sources.lock.json)。[Sparse4D 当前提交](https://github.com/HorizonRobotics/Sparse4D/commit/249ffbb695f4e9db628d953e2bf6d36de04bbb69)可直接核对两行 README 变更。当日 API 核验结果见[来源快照](../verification/host1_20261011/latest/source_heads.json)与[模型及数据快照](../verification/host1_20261011/latest/model_dataset_heads.json)。

## DrivingSDK 配套版

采用已发布、已取得官方 A3 镜像并实测的 **26.1.0** 分支提交 `0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837`，不是开发分支最新 HEAD。当日另观察到主线 `c968f92674be053312fff0528059098eec636928`、`branch_v26.2.0` 的 `d9d55f3e3126ac69d100fe5f980042f73093f935`；官方 registry 仍显示 26.1 / 26.0 配套镜像，未取得 26.2 发布镜像及本项目训练证据，因此未切换配套栈。

26.1 A3 镜像 ARM64 digest 本次复核仍为 `52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801`。固定发布分支与[官方安装文档](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/docs/zh/installation/installation.md)一起使用。

## Qwen 镜像更新及实测

`quay.io/ascend/llamafactory:latest-a3-openeuler` 已变化。新 ARM64 manifest 为 `f1aa74bf63e62cd9503f8b34922b6a26f14c49bf1910b484b44cd57d624cfad1`，镜像 ID 为 `5134fb97fcfe4a9a83103fbe45c6f196b847e4cfac2f873a29fb4effab8e7645`；本仓已将当前选择更新为 `local/llamafactory:a3-arm64-20261011`。

新镜像重新通过两物理卡 BF16 LoRA 10 步、保存参数与优化器检查、保存后单张独立图文推理；不是仅验证导入或启动。结果见[Qwen 保存检查](../verification/host1_20261011/qwen_latest/checkpoint_check.json)和[推理检查](../verification/host1_20261011/qwen_latest/inference_check.json)。旧 manifest 当日返回 404，已验证的旧镜像保留在交付归档中供回退。

五镜像角色、来源、实际 ID 和软件版本见 [images.lock.json](../images.lock.json)。当前训练只选择其中三个镜像；另外两个是官方基础镜像及旧 Qwen 回退镜像。实际版本组合保留兼容旧 OpenMMLab 的要求，未将所有 Python 包单独升级到各自最新版。
