# WorldEngine AlgEngine：Ascend 训练迁移

## 实现与范围

上游 [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine) 固定提交 `3eded7e8ac64e5bed817860765089b5914ab698f`。本目录覆盖 AlgEngine 的 `e2e_vadv2_50pct.py`：NAVFormer / TrajScoringHead、NPU 分布式入口及多尺度变形注意力。配置中的“50%”指上游数据划分；合成检查没有使用这些真实数据。SimEngine、3DGS、强化学习和完整后训练链路尚未适配。

## 镜像

1# 使用 `local/migration-worldengine:26.1.0-a3-20261011`，image ID `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3`。基于 DrivingSDK headless 镜像，保留 Python 3.10.21、CANN 9.1.0、PyTorch 2.7.1、TorchNPU 2.7.1.post6、MMCV 1.7.2、mmdet 2.24.0、mmdet3d 1.0.0rc4。

额外 nuPlan / AlgEngine 依赖隔离安装，nuPlan 使用维护方固定 Git 提交，NumPy 保持 1.23.5。ARM64 地理库来自独立 conda prefix。Dockerfile、依赖清单及来源见[环境指南](../../../../../docs/environment.md)。

## 快速开始

在已经准备好镜像和源码的 1#，从仓库根目录执行：

```bash
bash tools/quickstart.sh worldengine check
bash tools/quickstart.sh worldengine download
bash tools/quickstart.sh worldengine compute
```

`download` 只下载发布方 3932288 字节的 `test_8192_kmeans.npy`，SHA256 `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35`。`compute` 默认选四张不同物理卡的宿主逻辑设备 `0,2,4,6`，容器内编号 `0,1,2,3`，不加载真实训练数据或预训练 checkpoint。

```bash
bash tools/quickstart.sh worldengine build  # 仅模型构建和 NPU 放置
bash tools/quickstart.sh worldengine plan   # 列出真实训练缺少的文件
```

## 实机验证

2026-10-11 在 1# 上，最终镜像的环境导入、58,207,822 参数模型构建及四物理卡 DDP 合成输入检查通过。四帧、八相机 `96×160`，原配置 `200×200` BEV、完整 8192 条轨迹词表；各 rank 使用不同随机输入。四个 rank 均有有限的原始 BEV、规划输出、掩码处理前的 imitation loss 和 259 个梯度张量，AdamW 更新发生，更新后的被检查参数跨 rank 一致。

rank 0 的 checkpoint 为 233338823 字节，SHA256 `3b97ae39e728cde5fc0518a714a0d3a0b70fa9e04abd3702c55c1407be6b7288`，保存重载通过。报告位于 `outputs/worldengine/compute_fixture/rank{0,1,2,3}.json`，公开摘要见[实测记录](../../../../../docs/validation.md)。

这证明当前合成输入计算路径可运行。真实数据训练、正式输入分辨率、PDM / 开环规划指标、长期稳定性和 SimEngine 闭环链路未验证；完整训练验收仍未完成。

## 真实训练准备

按固定上游[数据指南](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/data_organization.md)准备 OpenScene/NAVSIM 图像、处理后的 navtrain/navtest 标注、两套 PDM 缓存、轨迹词表和基座 checkpoint。现有 `plan` 会检查关键路径，不能只准备词表就运行训练。

文件齐全后运行 `bash tools/quickstart.sh worldengine train`。4 卡、每卡 batch 1、学习率 `5e-5`、8 epoch 为本仓实验配置；需要重新评估全局 batch 和学习率。相应容器内的 `test/train_4p.sh --smoke --execute` 使用 1 epoch，并显式设置 `ASCEND_DEVICES=0,1,2,3`；大数据下 1 epoch 可能耗时较长。
