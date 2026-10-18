# 版本核验：后四项 2026-10-18；原五项保留历史日期

## 后四项最新源码与已验证环境

LSTM-Argoverse、STGCN-Traffic、ResWorld 的默认分支 HEAD 已实时刷新，与各自锁定提交一致。MobileNetV2 的 ModelZoo GitCode 主仓 HEAD 已前进至 `4f2041b142e599f79acfd0b5f68b2699d65b4f58`，其 MobileNetV2 子树与锁定提交 `e779117e8cfeba296a7105e26adac0741ab04768` 完全相同，tree 为 `75fd4c46d94f560f776c57f48a53dc705c682de3`。GitHub 镜像不作为 ModelZoo 主仓最新性的依据。

| 模型 | 已验证上游提交 | 核验材料 |
| --- | --- | --- |
| LSTM-Argoverse | `27696a19b3753a4ad8dcae3ad9c6107c5e4e237c` | [本模型来源核验](../PyTorch/contrib/cv/others/LSTM-Argoverse/validation/source_audit_20261018.json) |
| STGCN-Traffic | `5cfa82c78984fc58fafe4841c35618c6647c4219` | [本模型来源核验](../PyTorch/contrib/cv/others/STGCN-Traffic/validation/source_audit_20261018.json) |
| ResWorld | `a5ae973001dd6001d95b3389020a9e263d73a56e` | [本模型来源核验](../PyTorch/contrib/cv/others/ResWorld/validation/source_audit_20261018.json) |
| MobileNetV2 | `e779117e8cfeba296a7105e26adac0741ab04768`；当前主仓模型子树相同 | [本模型来源核验](../PyTorch/contrib/cv/classification/MobileNetV2/validation/source_audit_20261018.json) |

核验的准确 UTC 时间写在每模型 JSON。复现固定已经测试的源码和环境，没有自动升级依赖。后四项使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010`，精确 image ID 为 `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`；CANN9.1.0、Torch2.7.1+cpu、TorchNPU2.7.1.post6、mx_driving26.1.0.post2。镜像归档字节和 SHA 固定在各自 `images.lock.json`。

后四项当前检查点已包含本项目完整数据训练或扩展训练产物；MobileNetV2 官方 V1 和 ResWorld 作者最终权重分别作为预训练评估使用。不得把这些权重说成本项目完成了600epoch ImageNet或正式12epoch ResWorld训练。当前逐模型结果见[独立交付索引](additional_models_delivery.md)。

以下原五项是带日期的历史快照，不代表在2026-10-18重新完成了最新源码、数据或完整训练验收。

## 原五项的历史核验（2026-10-12）

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
