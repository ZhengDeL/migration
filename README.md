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

进入对应模型目录，按它自己的指导操作。每个目录包含独立的来源锁、镜像锁、补丁、数据准备与检查脚本，以及 `test/train_*p.sh`、`test/train_full_*p.sh` 和 `test/validate_1p.sh`。入口通过脚本位置确定模型目录，可单独复制到其他路径，不依赖迁移仓 Git 根目录。

BEVFormer / Sparse4D 采用华为 DrivingSDK 固定模型补丁及必要辅助文件；其余模型按固定官方上游实现适配。官方样例的默认八卡配置与本项目的两卡 / 四卡功能配置有所区别，具体修改和验收范围在各模型说明中列出。

## 交付和证据

- [独立模型离线交付](docs/offline_delivery.md)：每个模型独立目录及归档，单独取得即可使用。
- [社区调研](docs/community_research.md)：华为 / Ascend 已有实现与上游依据。
- [版本核验](docs/latest_versions.md)：固定来源、镜像及权重版本。
- [历史实测](docs/validation.md)：此前验证范围；新的独立入口证据分别放在各模型 `validation/`。

GitHub 提供源码和指导；完整模型交付另含对应镜像、数据、权重和保存结果。内部问题台账及操作记录不进入公开交付。

1# 实机为 Atlas 800I A3 推理系列，四卡测试也在该机进行。实验功能通过不改变华为官方训练硬件支持范围。当前使用 mini / 功能子集，完整数据精度、性能、长期稳定及 checkpoint 续训均需另行验收。
