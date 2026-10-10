# Apollo VisionNet：Ascend 训练迁移草案

## 参考实现与范围

上游是 [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net) 的 `4909aa681bc435a74e793e714b8066d9bbf0095d`。本目录补丁针对 `bev_tiny_det_occ_apollo.py` 的 BEV 检测与占用训练路径，修改训练入口、NPU 分布式包装和三处多尺度变形注意力调用；不涉及 TensorRT 部署仓。

## 环境和数据

**当前未选定、构建或在 2# 验证 Apollo VisionNet 训练镜像。** 上游依赖 Python 3.7、PyTorch 1.10、MMCV 1.4.1、MMDetection3D 0.18.1，尚未证明可直接运行在当前 TorchNPU A3 镜像。1# 的 DrivingSDK 派生镜像仅验证了 BEVFormer/Sparse4D，不能当作本模型的已验证镜像。需先在 2# 核对硬件和驱动，建立独立的 CANN/TorchNPU/MMCV/mmdet/mmdet3d 依赖矩阵，源码编译 NPU 算子，并执行 `python3 tools/check_env.py --model apollo_visionnet --nproc 4`、变形注意力前后向数值检查。镜像选择原则见仓库[环境指南](../../../../../docs/environment.md)。

按上游 [数据准备文档](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/main/docs/prepare_dataset.md)准备 nuScenes、CAN bus、OCC 标注和 train/val pkl；还需 `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只打印计划：

```bash
python3 tools/prepare.py apollo_visionnet --source sources/Apollo-Vision-Net --apply
bash PyTorch/contrib/cv/detection/Apollo-Vision-Net/test/train_4p.sh --smoke
bash PyTorch/contrib/cv/detection/Apollo-Vision-Net/test/train_4p.sh --smoke --execute
```

脚本以 4 卡、每卡 batch 1、`1e-4` 学习率为实验起点。`--smoke` 对 epoch runner 执行 1 个 epoch，可能耗时较长。

## 验证和状态

补丁已做逐文件静态校验、Python 编译和命令生成；旧版 OpenMMLab 兼容性、四卡训练、检测 mAP 与占用 mIoU 均未验证。上述指标通过前，迁移状态仍为草案。
