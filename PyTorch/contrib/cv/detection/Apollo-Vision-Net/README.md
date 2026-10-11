# Apollo VisionNet：Ascend 训练迁移草案

## 参考实现与范围

上游是 [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net) 的 `4909aa681bc435a74e793e714b8066d9bbf0095d`。本目录补丁针对 `bev_tiny_det_occ_apollo.py` 的 BEV 检测与占用训练路径，修改训练入口、NPU 分布式包装和三处多尺度变形注意力调用；不涉及 TensorRT 部署仓。

## 环境和数据

1# 实际使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010` 中的 `bevformer` 环境：Python 3.10.21、PyTorch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。已完成官方 DLA34/BEV+OCC 配置的源码导入、模型构建和单卡合成输入前后向检查；四卡和真实 OCC 数据训练仍待实测。该范围不等于旧版上游环境与完整训练已验收。镜像来源、ID 和包版本见仓库[环境指南](../../../../../docs/environment.md)。

按上游 [数据准备文档](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/prepare_dataset.md)准备 nuScenes、CAN bus、OCC 标注和 train/val pkl；还需 `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只打印计划：

```bash
python3 tools/prepare.py apollo_visionnet --source sources/Apollo-Vision-Net --apply
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet build
bash tools/quickstart.sh apollo_visionnet compute
bash tools/quickstart.sh apollo_visionnet plan
bash PyTorch/contrib/cv/detection/Apollo-Vision-Net/test/train_4p.sh --smoke
bash PyTorch/contrib/cv/detection/Apollo-Vision-Net/test/train_4p.sh --smoke --execute
```

脚本以 4 卡、每卡 batch 1、`1e-4` 学习率为实验起点。`--smoke` 对 epoch runner 执行 1 个 epoch，可能耗时较长。

## 验证和状态

`compute` 使用随机初始化和合成的三帧、六相机 `96×160` 输入，覆盖 DLA、时序 BEV、GroupDETR 检测和 OCC 损失。它不需要 nuScenes 或 OCC 归档，结果名称为 `pass_compute_fixture`，不代表真实数据训练。单卡实测有 465 个有限梯度张量，检测查询参数更新，原始检测/OCC 输出均为有限值；生成 282956419 字节 checkpoint。兼容补丁把 NumPy 角度显式转成 Python `float`，修复 torchvision 的角度类型报错。

四卡 DDP、正式分辨率、真实 OCC 数据、检测 mAP 和占用 mIoU 仍未验证；完整训练迁移状态保持草案。
