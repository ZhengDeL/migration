# Apollo VisionNet：Ascend 训练迁移草案

## 参考实现与范围

上游是 [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net) 的 `4909aa681bc435a74e793e714b8066d9bbf0095d`。本目录补丁针对 `bev_tiny_det_occ_apollo.py` 的 BEV 检测与占用训练路径，修改训练入口、NPU 分布式包装和三处多尺度变形注意力调用；不涉及 TensorRT 部署仓。

## 环境和数据

1# 实际使用 `local/migration-drivingsdk:26.1.0-a3-headless-20261010` 中的 `bevformer` 环境：Python 3.10.21、PyTorch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。已完成官方 DLA34/BEV+OCC 配置的源码导入、模型构建和四张物理卡 DDP 合成输入前后向检查。真实 OCC 数据训练和精度仍待验证。镜像来源、ID 和包版本见仓库[环境指南](../../../../../docs/environment.md)。

按上游 [数据准备文档](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/prepare_dataset.md)准备 nuScenes、CAN bus、OCC 标注和 train/val pkl；还需 `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`。

## 快速开始

在已准备好镜像和源码的 1# 上，从仓库根目录执行。`check` 检查环境，`build` 构建模型，`compute` 执行合成输入计算，`plan` 列出真实训练缺少的文件：

```bash
bash tools/quickstart.sh apollo_visionnet check
bash tools/quickstart.sh apollo_visionnet build
bash tools/quickstart.sh apollo_visionnet compute
bash tools/quickstart.sh apollo_visionnet plan
```

默认选宿主逻辑设备 `0,2,4,6`，在 1# 上对应四张不同物理卡，容器内编号为 `0,1,2,3`。如需改卡，先看 `npu-smi info`，再设置 `ASCEND_HOST_DEVICES`。真实数据和官方权重准备齐全后，使用 `bash tools/quickstart.sh apollo_visionnet train`；训练以 4 卡、每卡 batch 1、`1e-4` 学习率为实验起点。短程真实训练可在相应容器内用 `test/train_4p.sh --smoke --execute`，需要设置容器内 `ASCEND_DEVICES=0,1,2,3`，该模式运行 1 个 epoch。

## 验证和状态

`compute` 使用随机初始化和合成的三帧、六相机 `96×160` 输入，覆盖 DLA、时序 BEV、GroupDETR 检测和 OCC 损失。它不需要 nuScenes 或 OCC 归档，结果名称为 `pass_compute_fixture`。2026-10-11 在 1# 四张物理卡上实测正常退出，四个 rank 各有 465 个有限梯度张量，检测查询参数更新，原始检测/OCC 输出均为有限值；rank 0 生成 282956419 字节 checkpoint，并通过保存后的重载和有限值检查。SHA256 为 `18d2d3f0ab5d5913139ceee2865959b91b7e7b6937043139daea726ac5dc730c`。结果文件位于 `outputs/apollo_visionnet/compute_fixture/rank{0,1,2,3}.json`。

兼容补丁把 NumPy 角度显式转成 Python `float`，修复 torchvision 的角度类型报错。公开的四 rank JSON 见[实测记录](../../../../../docs/validation.md)。四卡合成输入检查证明当前计算路径可执行；正式分辨率、真实 OCC 数据、官方预训练初始化、检测 mAP 和占用 mIoU 仍未验证，完整训练迁移状态保持草案。
