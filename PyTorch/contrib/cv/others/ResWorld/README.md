# ResWorld：Ascend A3 独立迁移与验证

采用[作者固定源码](https://github.com/mengtan00/ResWorld/tree/a5ae973001dd6001d95b3389020a9e263d73a56e)，保留80,766,238参数、三时刻六相机、原深度与两项规划loss。发布前再次核对上游HEAD。

## 离线包快速上手

完整解压一个ResWorld.zip，进入本README目录，在ARM64 Linux Ascend服务器执行：

```bash
bash test/revalidate_packaged_1p.sh --devices 0
```

自动校验完整包、准备固定镜像，在新目录重载随包最终权重。镜像、源码、真实数据和权重均在本模型ZIP内。GitHub保存源码、指导和验证摘要。重载检查的具体范围与完整训练/数值复验命令见[本模型指南](FULL_VALIDATION.md)。

## 本轮实机结果

- 原156参数失败及后续1参数失败均已修复；五份固定权重各446个有效梯度全部通过原阈值。
- 第三项修复前后446个CPU参考梯度逐位相同；此前越界参数差由0.0025835降至0.0000926。
- 四卡10+10对20：2054/2054完整状态张量逐位相同，四rank实际增强输入一致。
- 四卡1811.968秒持续训练、最终权重数值与独立重载、作者全部65帧原规划/碰撞对照通过。

三处实际修改为官方erf GELU、等价稳定时序残差、固定常量几何投影CPU FP32路径。NN特征、采样和反向仍在NPU。根因、历史失败与CPU参考范围见[数值修复说明](NUMERICAL_REPAIR.md)。正式nuScenes trainval/12epoch及驾驶业务精度尚未验收。

## 固定环境

镜像：`local/migration-drivingsdk:26.1.0-a3-headless-20261010`。

精确image ID：`sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。

CANN9.1.0、Python3.10.21、Torch2.7.1+cpu、TorchNPU2.7.1.post6、DrivingSDK26.1.0.post2；MMCV1.7.2/MMDet2.24/MMDet3D1.0.0rc4。离线镜像归档、SHA和来源见 `images.lock.json`。宿主需要Docker、兼容Ascend驱动与空闲物理卡；实际板卡为Atlas800I A3推理系列，实测不改变官方训练支持范围。

## 自己重新训练

```bash
bash test/setup.sh
bash test/check_env.sh
bash test/train_4p.sh --devices 0,2,4,6 --steps 10 --output_dir "$PWD/test/output/train_new"
bash test/validate_1p.sh --devices 0 --output_dir "$PWD/test/output/train_new"
```

每物理卡只选一个宿主逻辑ID；0,2,4,6代表四张不同物理卡。每次使用新输出目录。精确恢复、五份权重数值检查、吞吐与30分钟脚本均独立放在本模型目录。旧数值profile不能直接作为新profile精确续训。
