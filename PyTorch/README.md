# PyTorch 训练迁移目录

分类沿用 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的 `contrib/cv/detection`、`contrib/cv/others`、`contrib/nlp` 路径。每个模型目录包含 README 和 `test/` 启动脚本；自编迁移补丁或训练配置与模型放在同一目录。共享脚本和版本锁定见仓库根目录。

| 类别 | 模型 | 训练卡数 | 当前阶段 |
| --- | --- | ---: | --- |
| cv/detection | [BEVFormer](contrib/cv/detection/BEVFormer/README.md) | 2 | 官方补丁本地验证 |
| cv/detection | [Sparse4D](contrib/cv/detection/Sparse4D/README.md) | 2 | 官方补丁本地验证 |
| cv/detection | [Apollo-Vision-Net](contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 | 应用层补丁草案 |
| cv/others | [WorldEngine](contrib/cv/others/WorldEngine/README.md) | 4 | AlgEngine 应用层补丁草案 |
| nlp | [Qwen3-VL-8B](contrib/nlp/Qwen3-VL-8B/README.md) | 2 | LoRA 配置本地验证 |

所有模型的 910C 实机训练、性能和精度结果均待验证；详细门槛见仓库根目录 README。
