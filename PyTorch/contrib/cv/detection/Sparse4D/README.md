# Sparse4Dv3：Ascend 训练迁移

## 参考实现与范围

上游是 [HorizonRobotics/Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 的 `c41df4bbf7bc82490f11ff55173abfcb3fb91425`。复用 [DrivingSDK Sparse4D 样例](https://gitcode.com/Ascend/DrivingSDK/blob/branch_v26.1.0/model_examples/Sparse4D/README.md) 的模型和依赖补丁。`tools/prepare.py` 还会复制并核验官方 `tools/patch.py`。

## 环境和数据

使用隔离的 DrivingSDK/Sparse4D 环境，按官方样例构建 NPU 版 MMCV、mmdet 并安装 mx_driving。训练前执行 `python3 tools/check_env.py --model sparse4d --nproc 2`。需准备 nuScenes、`data/nuscenes_anno_pkls/nuscenes_infos_{train,val}.pkl`、`nuscenes_kmeans900.npy` 和 `ckpt/resnet50-19c8e357.pth`；数据制作按官方样例执行。

## 快速开始

以下命令从仓库根目录执行；未传 `--execute` 时只打印计划：

```bash
python3 tools/prepare.py sparse4d --source sources/Sparse4D --sdk sources/DrivingSDK --apply
bash PyTorch/contrib/cv/detection/Sparse4D/test/train_2p.sh --smoke
bash PyTorch/contrib/cv/detection/Sparse4D/test/train_2p.sh --smoke --execute
```

启动器由官方配置生成输出目录中的副本，把 `total_batch_size` 48→2、`num_gpus` 8→2、学习率 `6e-4`→`2.5e-5`，同时按 2 卡重新计算每 epoch 迭代数。`--smoke` 只跑 10 次优化迭代，不覆盖上游配置。

## 验证和状态

本地已验证补丁逐文件内容、配置生成、Python 编译和命令生成；910C 训练尚未执行。实机需核对两 rank 前后向、有限 loss、checkpoint 和 nuScenes mAP/NDS，学习率以精度结果确认。
