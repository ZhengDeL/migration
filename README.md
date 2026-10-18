# Ascend 910C/A3 模型迁移

按 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的类别和模型目录组织。每个模型独立提供来源与镜像锁、环境/数据准备、训练脚本、验证脚本和实际测试数据。

## 离线模型包下载

- 分享名称：`ascend-a3-9models-20261018`
- 百度网盘：[下载九个独立模型包](https://pan.baidu.com/s/1jklwz-kzC1QQfohh1STDqQ?pwd=t8qj)
- 提取码：`t8qj`

共九个独立模型包。每个模型包包含自己的**指导、源码及适配、运行镜像、已准备的数据和权重、训练与验证脚本、检查点及测试结果**。下载所需模型的 ZIP，完整解压后按包内 README 使用。

## 选择模型

| 独立模型指导 | 训练物理卡数 | 本仓最新实机记录 |
| --- | --- | --- |
| [BEVFormer](PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2 | 2026-10-12 |
| [Sparse4D](PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2 | 2026-10-12 |
| [Qwen3-VL-8B](PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2 | 2026-10-12 |
| [WorldEngine](PyTorch/contrib/cv/others/WorldEngine/README.md) | 4 | 2026-10-12 |
| [Apollo-Vision-Net](PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 | 2026-10-12 |
| [LSTM-Argoverse](PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 2 | 2026-10-18 |
| [STGCN-Traffic](PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 2 | 2026-10-18 |
| [ResWorld](PyTorch/contrib/cv/others/ResWorld/README.md) | 4 | 2026-10-18 |
| [MobileNetV2](PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 2 | 2026-10-18 |

进入对应模型目录，按该模型 README 的“准备训练环境 → 开始训练 → 训练结果展示”执行。各模型可单独部署，训练/验证入口不依赖总仓 Git 根目录。完整训练协议及支持范围以每个模型的指导为准。

## 文档

- [九模型验证结果](docs/validation.md)：数据集、划分、更新次数、任务指标、数值误差、结果 JSON 及未完成项。
- [环境与镜像](docs/environment.md)：已验证的三个镜像组合与 image ID。
- [独立离线交付](docs/offline_delivery.md)：每模型完整 ZIP 的内容、校验和使用方式。
- [固定版本与核验日期](docs/latest_versions.md)：各模型来源提交及实际核验依据。
- [社区迁移依据](docs/community_research.md) / [原仓指导格式对应](docs/readme_reference_alignment.md)：上游实现、华为社区适配与原始说明。

## 目录

```text
PyTorch/contrib/     按任务分类的九套独立模型工程
docs/               结果汇总、环境、交付和来源说明
docker/             已验证镜像的构建文件与固定依赖
```

模型目录中的 `test/` 为操作入口，`tools/` 为适配实现，`validation/` 为可公开测试依据。已随仓保存的固定上游文件位于模型 `vendor/`；大镜像、数据、权重和检查点由各模型完整 ZIP 提供，详见 [第三方来源说明](NOTICE.md)。

实机为 Atlas 800I A3 推理系列。LSTM 与 STGCN 完成了各自注明的完整参考训练；ResWorld 正式 trainval/12 epoch、MobileNetV2 从头 600 epoch 和原五项完整数据精度仍有未完成项，见[验证总表](docs/validation.md)。
