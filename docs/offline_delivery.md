# 独立模型离线交付

每个模型单独交付一个完整 ZIP。取得所需模型 ZIP，校验外部 SHA256 清单，完整解压，再按其中 README 执行。每个模型拥有自己的运行镜像、代码、数据、权重、训练与验证入口，可独立使用。

## 下载地址

- 分享名称：`ascend-a3-9models-20261018`
- 百度网盘：[下载九个独立模型包](https://pan.baidu.com/s/1jklwz-kzC1QQfohh1STDqQ?pwd=t8qj)
- 提取码：`t8qj`

共九个独立模型包。每个模型包包含自己的**指导、源码及适配、运行镜像、已准备的数据和权重、训练与验证脚本、检查点及测试结果**。各包独立使用，所含数据与已验证范围见下表和对应模型指导。

该分享为 **2026-10-18 固定交付版本**。GitHub 于 2026-10-19 新增的三模型全量复验、Qwen 完整 DriveLM 准备及公开入口实测，以及正在准备的其他完整数据尚未重新打入这批 ZIP。包内旧记录与当天新结果分别保留；后续完整验收和重新打包完成前，不将该分享称为八模型完整数据验收包。WorldEngine AlgEngine 后续使用[独立迁移仓](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine)。

## 各模型入口

| 模型指导 | 数据与权重范围 |
| --- | --- |
| [BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md) | 功能数据 / 本模型所需初始化与产物 |
| [Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md) | 功能数据 / 本模型所需初始化与产物 |
| [Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 功能数据 / 官方权重与本项目产物 |
| [WorldEngine](../PyTorch/contrib/cv/others/WorldEngine/README.md) | 功能数据 / 官方权重与本项目产物 |
| [Apollo-Vision-Net](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 功能数据 / 官方权重与本项目产物 |
| [LSTM-Argoverse](../PyTorch/contrib/cv/others/LSTM-Argoverse/README.md) | 完整参考训练数据 / 本模型所需初始化与产物 |
| [STGCN-Traffic](../PyTorch/contrib/cv/others/STGCN-Traffic/README.md) | 完整参考训练数据 / 本模型所需初始化与产物 |
| [ResWorld](../PyTorch/contrib/cv/others/ResWorld/README.md) | 功能数据 / 官方权重与本项目产物 |
| [MobileNetV2](../PyTorch/contrib/cv/classification/MobileNetV2/README.md) | 功能数据 / 官方 ImageNet 全 val |

## 每个 ZIP 包含什么

- 本模型 README、适配源码、固定来源/镜像/资产锁和所需上游文件。
- 一份 `images/` 运行镜像归档；导入后按本模型 `images.lock.json` 核对 image ID。
- 本模型已准备的数据与初始化权重，以及已验证的参考检查点；范围以模型说明为准。
- 独立的 `test/` 训练/验证入口及 `tools/` 实现。
- `validation/` 测试依据、`package_manifest.json` 与 `SHA256SUMS` 文件校验清单。

## 工程师使用顺序

1. 校验 ZIP 外部 SHA256，完整解压到 ARM64 Ascend Linux 服务器。Windows 可查看说明和校验归档。
2. 进入这个模型目录，按 README 校验包并执行 setup，导入镜像和准备路径。
3. 检查环境、物理卡号和数据计划。功能子集与 full 模式分别按指导操作。
4. 使用本模型 `test/train_*p.sh` 训练，运行本模型独立验证入口；后四项还提供随包产物复验入口，详见各自指导。
5. 检查本次退出码、报告和检查点身份，对照[验证总表](validation.md)。`test/output/` 保存本次输出，参考结果与新结果按运行目录区分。

LSTM 包含完整 train/val 原始归档和处理缓存；MobileNetV2 提供完整 ImageNet val 与 V1 权重，其功能训练仍使用 Imagenette。ResWorld 提供真实 mini 和所需作者权重，正式 trainval/12 epoch 尚未完成。已准备的完整数据不能代表所有训练协议均已通过。

网盘交付范围为九个独立模型 ZIP 与必要说明/校验清单。后四项的独立 Word 迁移报告仅保留在本地，不纳入网盘上传内容。交付包不收录本地问题台账、服务排查日志、临时下载或 SSH 辅助程序。

GitHub 文档可随整理持续更新；离线 ZIP 内文档与内容以该 ZIP 清单为准。只有重新打包、校验并复验后才产生新的归档版本，不能用 GitHub 新提交号推断旧 ZIP 已更新。
