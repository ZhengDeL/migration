# 新增四个智驾相关模型：实测与独立交付

## 独立工程

| 模型 | 使用指导 | 实际物理卡 | 真实数据验证范围 |
| --- | --- | ---: | --- |
| LSTM-Argoverse | [独立 README](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 2 | 官方轨迹 train/val 子集 128/32，10 步训练、重载、CPU 对照、ADE/FDE/Miss Rate |
| STGCN-Traffic | [独立 README](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 2 | PeMSD7(M)，10 步训练、重载、CPU 对照、241 val 窗口 |
| ResWorld | [独立 README](../PyTorch/contrib/cv/others/ResWorld/README.md) | 4 | 真实 nuScenes mini/CAN/深度，完整 80M 网络，10 步训练、重载、CPU 对照、65 val 帧 |
| MobileNetV2 | [独立 README](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 2 | Imagenette 100 train/40 val，1000 类头，10 步训练、重载、CPU 对照 |

以上均已在 1# 实际执行，训练每个 rank 都保留损失、梯度、参数更新和全参数 SHA256 同步证据。每项的 checkpoint 与任务指标独立记录。真实数据子集与随机初始化的短训练用于功能验证，正式精度、完整训练、性能和长稳均未验收。

## 离线文件

每个模型交付一份独立 ZIP，并保留对应的可直接使用的展开目录。单独取得一个模型 ZIP 即可获得它所需的指导、源码、镜像、真实功能数据、已验证 checkpoint、重载参考、必要验证 JSON 及 SHA256 清单；运行不依赖其他模型目录。

解压后，进入该模型 README 所在目录，按该 README 的命令操作。大镜像、数据和 checkpoint 不进入 GitHub；GitHub 提供版本固定的代码、指导和去私有化的必要验证结果。网盘由用户自行上传。

本地问题记录、诊断日志、临时下载和 SSH helper 均不属于公开交付。下载缓存内的完整 Argoverse 归档未加入 LSTM 功能包，ResWorld 冗余原始 expansion ZIP 也未重复打包；每项只保留当前独立复验所需内容和来源证据。

## 版本锁和限制

2026-10-13 对作者 LSTM / STGCN / ResWorld 默认分支及 ModelZoo GitCode 主仓 HEAD 再次核验，均与本次固定提交一致。ModelZoo GitHub 镜像停在旧版，主仓核验使用 GitCode。镜像使用已经实测并固定 image ID 的 CANN / TorchNPU / DrivingSDK 配套，不随 tag 自动升级。核验结果随各模型独立 `validation/upstream_status.json` 提供。

Atlas 800I A3 推理系列上的成功实验不改变华为官方训练硬件支持范围。当前包没有声称复现论文精度；LSTM curriculum 和 ResWorld 正式预训练 / EMA / SyncBN / warmup / 碰撞指标仍需另行验收。
