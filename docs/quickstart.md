# 模型指南入口

选择对应模型，进入它自己的目录，按该模型 README 操作。每个模型拥有独立的来源锁、镜像锁、数据准备、训练及验证脚本；完整操作步骤写在各模型目录中。

| 模型 | 独立指导 | 功能训练入口 | 独立验证入口 |
| --- | --- | --- | --- |
| BEVFormer | [README](../PyTorch/contrib/cv/detection/BEVFormer/README.md) | `test/train_2p.sh` | `test/validate_1p.sh` |
| Sparse4D | [README](../PyTorch/contrib/cv/detection/Sparse4D/README.md) | `test/train_2p.sh` | `test/validate_1p.sh` |
| Qwen3-VL-8B | [README](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | `test/train_2p.sh` | `test/validate_1p.sh` |
| WorldEngine AlgEngine | [README](../PyTorch/contrib/cv/others/WorldEngine/README.md) | `test/train_4p.sh` | `test/validate_1p.sh` |
| Apollo VisionNet | [README](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | `test/train_4p.sh` | `test/validate_1p.sh` |

GitHub 保存工程源码；模型完整离线目录另含其固定镜像、原始数据、初始化权重、保存结果和校验清单。单独取得某个模型即可使用，不需要其他模型目录或总仓库的 Git 元数据。
