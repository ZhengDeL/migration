# Ascend 910C/A3 模型迁移

按 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的类别和模型目录组织。每个模型是一套独立工程，自己的 README 给出环境、镜像、数据、训练、独立验证、成功判据和常见问题。

## 选择模型

| 模型 | 独立指导与代码 | 训练物理卡数 |
| --- | --- | ---: |
| BEVFormer | [cv/detection/BEVFormer](PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2 |
| Sparse4Dv3 | [cv/detection/Sparse4D](PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2 |
| Qwen3-VL-8B-Instruct | [nlp/Qwen3-VL-8B](PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2 |
| WorldEngine AlgEngine | [cv/others/WorldEngine](PyTorch/contrib/cv/others/WorldEngine/README.md) | 4 |
| Apollo VisionNet | [cv/detection/Apollo-Vision-Net](PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 |
| LSTM 车辆轨迹预测 | [cv/others/LSTM-Argoverse](PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 2 |
| 交通 STGCN | [cv/others/STGCN-Traffic](PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 2 |
| ResWorld | [cv/others/ResWorld](PyTorch/contrib/cv/others/ResWorld/README.md) | 4 |
| MobileNetV2 | [cv/classification/MobileNetV2](PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 2 |

进入对应模型目录，按它自己的指导操作。每个目录包含独立的来源锁、镜像锁、适配代码、数据准备与检查脚本，以及 `test/train_*p.sh` 和 `test/validate_1p.sh`。全量训练协议和入口的支持范围以各模型 README 为准。入口通过脚本位置确定模型目录，可单独复制到其他路径，不依赖迁移仓 Git 根目录。

BEVFormer / Sparse4D 采用华为 DrivingSDK 固定模型补丁及必要辅助文件；其余模型按固定官方上游实现适配。官方样例的默认八卡配置与本项目的两卡 / 四卡功能配置有所区别，具体修改和验收范围在各模型说明中列出。

## 交付和证据

- [独立模型离线交付](docs/offline_delivery.md)：每个模型独立目录及归档，单独取得即可使用。
- [新增四模型实测与交付](docs/additional_models_delivery.md)：逐模型独立训练、重载、真实留出验证及离线包说明。
- [社区调研](docs/community_research.md)：华为 / Ascend 已有实现与上游依据。
- [新增智驾模型研究与进展](docs/additional_models_research.md)：LSTM 轨迹预测、交通 STGCN、ResWorld 与 MobileNetV2；逐项列出当前验证范围。
- [LSTM 智驾选型研究](docs/lstm_driving_research.md)：官方基础 LSTM 的任务、数据协议、固定来源及单 NPU 神经网络复验。
- [版本核验](docs/latest_versions.md)：固定来源、镜像及权重版本。
- [历史实测](docs/validation.md)：此前验证范围；新的独立入口证据分别放在各模型 `validation/`。

GitHub 提供源码和指导；完整模型交付另含对应镜像、数据、权重和保存结果。内部问题台账及操作记录不进入公开交付。

后四项已在用户新增机器执行 2026-10-18 完整或扩展验证：LSTM-Argoverse 完整三阶段训练，STGCN-Traffic 完整 PeMSD7(M) 训练，MobileNetV2 官方 ImageNet 五万图验证及功能训练，ResWorld 真实 mini 四卡与原作者权重规划/碰撞控制。各模型的数值通过、失败及未完成范围见自己的 FULL_VALIDATION.md 和 validation/extended_acceptance_20261018.json。ResWorld 原数值失败已修复：五份固定权重梯度、四卡严格恢复、30分钟及原65帧通过；正式trainval/12epoch仍未完成。MobileNetV2 从头完整训练、ResWorld 正式 trainval/完整训练协议尚未完成。实测设备为 Atlas 800I A3 推理系列，实验不改变官方训练支持范围。原五项保留各自带日期的历史记录。
