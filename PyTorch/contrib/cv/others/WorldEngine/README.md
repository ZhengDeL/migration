# WorldEngine AlgEngine：Ascend 训练迁移草案

## 参考实现与范围

上游是 [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine) 的 `3eded7e8ac64e5bed817860765089b5914ab698f`。本目录补丁覆盖 **AlgEngine VADv2 50%** 配置的训练入口、NPU 分布式包装及 UniAD 路径的多尺度变形注意力。SimEngine、3DGS 和完整后训练链路不在本补丁范围内。

## 环境和数据

上游安装指南依赖旧版 CUDA/OpenMMLab，尚未证明可与当前 TorchNPU A3 环境兼容。必须先建立独立的 MMCV/mmdet/mmdet3d 依赖矩阵、源码编译、`python3 tools/check_env.py --model worldengine --nproc 4` 和算子数值检查；不能把本地静态检查当成环境通过。

按上游 [AlgEngine 数据说明](https://github.com/OpenDriveLab/WorldEngine/blob/main/docs/algengine_usage.md)准备 OpenScene/NAVSIM 数据、`nuplan_openscene_nav{train,test}.pkl`、`test_8192_kmeans.npy` 和配置引用的基座 checkpoint。数据路径详见根目录 `docs/environment.md`。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只打印计划：

```bash
python3 tools/prepare.py worldengine --source sources/WorldEngine --apply
bash PyTorch/contrib/cv/others/WorldEngine/test/train_4p.sh --smoke
bash PyTorch/contrib/cv/others/WorldEngine/test/train_4p.sh --smoke --execute
```

脚本以 4 卡、每卡 batch 1、`5e-5` 学习率为实验起点。`--smoke` 对 epoch runner 执行 1 个 epoch，可能耗时较长。

## 验证和状态

补丁已做逐文件静态校验、Python 编译和命令生成；NPU 依赖栈、四卡训练、开环指标与 SimEngine 闭环链路均未验证。实机先核对变形注意力前后向，再做模型级精度回归。
