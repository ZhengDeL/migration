# 版本核验（原五项 2026-10-12；新增四项 2026-10-13）

本次独立交付使用当前核验的上游提交、配套官方发布镜像和完整权重；具体 image ID 和归档 SHA 固定在每个模型自己的 images.lock.json。后续滚动标签变化时需重新验证后再更新交付。

新增 LSTM / STGCN / ResWorld / MobileNetV2 使用各自 `validation/upstream_status.json` 记录的 2026-10-13 默认分支或 canonical HEAD 核验。LSTM `27696a19`、STGCN PyTorch `5cfa82c7`、STGCN 作者 `4ab75ab6`、ResWorld `a5ae9730`、ModelZoo GitCode `e779117e` 均与此次核验一致。ModelZoo 的 GitHub 镜像停在旧版，采用 GitCode 主仓核验结果。新增四项使用经过实测的 DrivingSDK 26.1 A3 headless 派生镜像，保持原始 26.1 发布配套，不切换尚未完成本项目训练验证的开发分支。

新增四项的 checkpoint 是本项目随机初始化短训练保存的功能产物，未被描述为官方完整预训练或精度达标权重。模型、任务和指标范围见 [新增四项实测](additional_models_delivery.md)。

## 上游和数据

| 项目 | 固定提交 | 2026-10-12 核验 |
| --- | --- | --- |
| BEVFormer | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | git ls-remote 与官方 HEAD 一致 |
| Sparse4D | `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | git ls-remote 与官方 HEAD 一致 |
| LLaMA-Factory | `1fded918ac4723ac9966363f51ed4500866c5d1d` | git ls-remote 与官方 HEAD 一致 |
| WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` | git ls-remote 与官方 HEAD 一致 |
| Apollo VisionNet | `4909aa681bc435a74e793e714b8066d9bbf0095d` | git ls-remote 与官方 HEAD 一致 |
| Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | [官方 main 文件树](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main)及最新提交链接一致；四权重分片 SHA 固定 |
| WorldEngine 数据 | `d0e88d858533ca407a3387a26577c083aa7b3617` | [官方 main 文件树](https://huggingface.co/datasets/OpenDriveLab/WorldEngine/tree/main)及最新提交链接一致 |

BEVFormer / Sparse4D 使用 DrivingSDK 发布分支 `branch_v26.1.0` 的当前提交 `0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837`。当天主线为 `c264ed252fae3e36b2fbbbe1d5f3ea1b67cf2324`，26.2 开发分支为 `d9d55f3e3126ac69d100fe5f980042f73093f935`；本次选择与实际镜像配套、已验证的 26.1 发布分支。

## 官方镜像与实际版本

DrivingSDK 26.1 A3 的当前 ARM64 registry digest 仍为 `52326fada68feb9d26e8c170bb7fdeea0e81aeba24fa1a84a316a19feba3c801`。BEVFormer / Sparse4D / Apollo 使用其 headless 派生镜像；WorldEngine 使用加装地理依赖的派生镜像。每个模型目录只包含自己的运行镜像归档。

Qwen 官方 `latest-a3-openeuler` 已更新。2026-10-12 当前 ARM64 digest 为 `6394a17532563a9b978b2e524a20594287b8ee480fe159e4e57b5067ec8125ea`，实际 image ID 为 `99d33965ae4502b07bffe2c3248887795ae0672a0e79dd0aaa567d09cce811b5`，交付标签 `local/llamafactory:a3-arm64-20261012`。

新旧镜像的文件层不同，主要软件版本保持 CANN9.1、Python3.12.13、Torch2.10、TorchNPU2.10.post2、transformers5.8。因此本次重新执行新镜像双卡 LoRA 十步训练、保存检查和独立图文推理，并验证单镜像导入及独立目录部署。当前 Qwen 指导、锁文件和验证结果均对应新 image ID；此前镜像及产物留在本机历史目录。

核验依据见[当前 registry](../verification/host1_20261012_independent/current_registry_versions.json)、[权重与数据](../verification/host1_20261012_independent/current_model_dataset_heads.json)和[Qwen 新旧镜像比较](../verification/host1_20261012_independent/qwen_current_image_comparison.json)。每个模型 `validation/` 给出本模型实际执行时间、保存检查与路径迁移验证。

软件版本按模型兼容组合固定。完整训练精度、长期稳定性和正式训练硬件支持另行验收；Atlas800I A3 的实验成功不改变官方训练支持范围。
