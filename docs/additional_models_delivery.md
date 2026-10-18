# 后四个模型：独立指导与实际验证（2026-10-18）

每个模型独立运行、独立记录结果，并单独提供 Word 迁移报告。先进入对应模型目录，使用自己的指导和训练脚本。

| 模型 | 独立指导 | 本轮实际覆盖 | 完成边界 |
| --- | --- | --- | --- |
| LSTM-Argoverse | [README](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) / [完整指导](../PyTorch/contrib/cv/others/LSTM-Argoverse/FULL_VALIDATION.md) | 全部205942 train / 39472 val；原1/10/30三阶段，725epoch、292175更新；完整预测、最终权重梯度、成熟状态断点恢复及独立重载 | val用于选模；不作为独立test、排行榜或驾驶业务精度验收 |
| STGCN-Traffic | [README](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) / [完整指导](../PyTorch/contrib/cv/others/STGCN-Traffic/FULL_VALIDATION.md) | 完整PeMSD7(M)、52epoch/14404更新；完整val/test、恢复、梯度、1p/2p吞吐、双卡30分钟与重载 | 对应选定实现70/15/15协议；METR-LA及其他PeMS未验收 |
| ResWorld | [README](../PyTorch/contrib/cv/others/ResWorld/README.md) / [独立指导](../PyTorch/contrib/cv/others/ResWorld/FULL_VALIDATION.md) | 完整80M网络、真实mini/CAN/深度，四卡训练、持续运行、恢复/梯度数值控制；作者完整权重原65帧规划/碰撞 | 原数值失败已修复，五份权重梯度、四卡严格恢复、30分钟和原65帧通过；正式trainval/12epoch未完成 |
| MobileNetV2 | [README](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) / [独立指导](../PyTorch/contrib/cv/classification/MobileNetV2/FULL_VALIDATION.md) | Imagenette功能训练、恢复、梯度、1p/2p吞吐、双卡30分钟与重载；官方V1权重全50000图/50000000 logits对照 | ImageNet Top1=71.87%、Top5=90.294%；600epoch从头训练尚未完成 |

## 使用与交付

GitHub 提供固定版本源码、每模型指导与必要验证摘要；完整独立 ZIP 另含本模型自己的镜像、数据、所需权重、已验证检查点、参考和 SHA256 清单。单独取得一个模型 ZIP，完整解压后即可按其 README 使用；不依赖其他模型目录。

每个模型新增 `test/revalidate_packaged_1p.sh`：一条命令校验包、准备镜像，在新输出目录独立重载随包检查点。各自的训练入口和完整/功能模式仍按各模型指导操作。完整训练状态与选择后的推理权重、作者预训练权重分别标明用途。

LSTM 包同时包含官方完整 train/val 原始压缩归档及完整处理缓存；MobileNetV2 包含官方完整 val.tar、devkit、V1 权重及索引；ResWorld 包保留真实 mini 与两种作者权重。未完成的 ImageNet train 下载分块不入包。镜像固定 image ID，不能用同名滚动 tag 代替已验证镜像。

四份 Word 分别说明每个模型的迁移代码、原始协议、环境镜像、数据来源、失败及修复、真实测试、检查点 SHA、最终归档与工程师复现步骤。它们不是一篇合并报告。最终上传目录只放核心模型 ZIP、Word ZIP 和清单/说明，由用户自行上传百度网盘。

本地问题台账、原始诊断日志、临时下载、SSH 辅助程序和凭据不进入公开仓库或核心上传包。推理系列设备上的实际训练实验不改变华为官方训练硬件支持范围。目录内的历史十步结果保留日期，新结论使用各自 `validation/extended_acceptance_20261018.json`。
