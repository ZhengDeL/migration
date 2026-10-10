# BEVFormer-base：Ascend 训练迁移

## 参考实现与范围

上游是 [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer) 的 `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`。复用 [DrivingSDK BEVFormer 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/BEVFormer/README.md) 的 NPU 补丁，以 FP16 base 配置做 2 卡训练。`sources.lock.json` 固定了上游、DrivingSDK 和补丁哈希。

## 环境和数据

在隔离的 DrivingSDK/BEVFormer 环境中安装配套 CANN、PyTorch、TorchNPU、mx_driving，以及按官方补丁源码构建的 MMCV、mmdet、mmdet3d。训练前执行 `python3 tools/check_env.py --model bevformer --nproc 2`。版本配套与镜像候选见仓库 `docs/environment.md`。

模型源码下需有 nuScenes v1.0 full、CAN bus、`data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl`、`ckpts/r101_dcn_fcos3d_pretrain.pth`。请按 DrivingSDK 样例生成标注；数据和权重不进入 Git。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只检查并打印启动计划：

```bash
python3 tools/prepare.py bevformer --source sources/BEVFormer --sdk sources/DrivingSDK --apply
bash PyTorch/contrib/cv/detection/BEVFormer/test/train_2p.sh --smoke
bash PyTorch/contrib/cv/detection/BEVFormer/test/train_2p.sh --smoke --execute
```

FP16 base 的 8 卡基线学习率为 `2e-4`；2 卡且每卡 batch 1 时以 `5e-5` 为线性缩放起点。`--smoke` 对 epoch runner 执行 1 个 epoch 并跳过验证，可能需要较长时间。

## 验证和状态

本地已验证补丁逐文件内容、Python 编译和命令生成；910C 训练尚未执行。实机需核对算子前后向、两 rank 有限 loss、checkpoint，以及 nuScenes mAP/NDS。学习率和精度需由实训确定。
