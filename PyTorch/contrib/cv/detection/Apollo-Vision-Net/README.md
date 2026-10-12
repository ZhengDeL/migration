# Apollo VisionNet：Ascend 独立训练与验证

本目录是一套单模型工程。使用本模型自己的源码、镜像、数据、训练脚本和验证脚本；可单独复制或解压到服务器，目录名与上级路径可改变。无需保留其他模型目录或迁移仓 Git 根目录。

## 1. 官方依据与环境

- [官方数据/项目说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/prepare_dataset.md)；[昇腾样例或官方训练说明](https://github.com/ApolloAuto/Apollo-Vision-Net/blob/4909aa681bc435a74e793e714b8066d9bbf0095d/docs/run.md)。
- 上游实现：https://github.com/ApolloAuto/Apollo-Vision-Net.git，固定提交 `4909aa681bc435a74e793e714b8066d9bbf0095d`。
- 本模型镜像：`local/migration-drivingsdk:26.1.0-a3-headless-20261010`，image ID `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2`。原始 registry digest 与软件版本见本目录 `images.lock.json`。
- 服务器：ARM64 Linux、配套 Ascend 驱动、Docker、Python 3.9+，训练需要 4 张空闲物理卡；模型 Python 与运行库在镜像内。`+cpu` 包版本标记不表示使用 CPU 训练，实际设备由环境检查和实测报告确认。
- 本次 1# 为 Atlas 800I A3 推理系列，实验训练结果不改变华为官方训练硬件支持范围。

## 2. 获取并准备本模型目录

取得本模型完整离线包并解压，进入包含本 README 的目录。包内镜像为一个独立 Docker 归档；原始数据、源码和权重均属于本模型，数据准备不访问网络。

```bash
cd Apollo-Vision-Net
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 3. 环境检查、训练与独立验证

```bash
npu-smi info
bash test/check_env.sh --device_id=0,2,4,6
bash test/plan.sh
bash test/train_4p.sh --device_id=0,2,4,6
bash test/validate_1p.sh --device_id=0
```

1# 一张物理卡有两个逻辑芯片；默认 `0,2,4,6` 每物理卡只取一个。容器内重新编号为连续 ID。换机先核对实际拓扑，换卡修改 `--device_id`。验证只用一张卡，只填一个 ID。

每次训练创建独立的 `test/output/functional_<时间>/`，保留已有结果；`test/output/latest_run.json` 指向最近一次尝试。训练失败时验证脚本拒绝把旧结果当成本次通过。所有原始终端输出、配置、checkpoint 和 JSON 报告均保存于本模型目录内。

本次功能配置：**官方 DLA 222 张量初始化；242 train / 162 独立 val；4 卡 FP32 每卡 batch1，学习率1e-4，10 步**。输出 `iter_10.pth`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 4. 通过判据

| 检查 | 判断方法 |
| --- | --- |
| 包完整性 | verify_package 退出0、文件大小与 SHA 匹配 |
| 环境 | `test/output/environment.json` 为 pass 或 pass_with_warnings，failures 为空；管理面告警须结合宿主 npu-smi 与实际计算判定 |
| 数据计划 | plan 的 missing_data 必须为 []；仅有文件还不代表训练通过 |
| 功能训练 | 命令退出0，当前运行目录 `train_result.json` 为 pass_functional_training；保存内容检查通过 |
| 独立验证 | validate 退出0，当前运行目录 `validation_result.json` 为 pass_independent_functional_validation |

验证内容：四 rank 输出/loss/465 梯度、更新、抽查同步及保存重载；严格加载后全部162独立验证帧推理有限。 每次新训练文件 SHA 可以不同，验收依据为对应本次运行的记录和检查内容。

未计算检测 mAP 或 OCC mIoU；完整归档结构检查与 mini404标签语义检查分别记录。 完整数据精度、性能、长期稳定性和 checkpoint 继续训练另行验收；不得把功能验证结果写成正式指标已通过。

## 5. 数据布局与从零准备

```text
data/nuscenes/{samples,sweeps,maps,v1.0-mini}
data/apollo/nuscenes_infos_temporal_{train,val}_occ_gt.pkl
data/apollo/occ_gt_release_v1_0.tar.gz 或已展开目录
sources/Apollo-Vision-Net/ckpts/官方DLA深度预训练权重.pth
```

`sources.lock.json` 固定本模型源码和补丁。本模型的NPU与兼容补丁保存在本目录，来源和SHA见sources.lock.json。 迁移到新目录后重新运行本模型的 `bash test/prepare_data.sh`。

仅源码部署时先在本模型的固定镜像中运行 `tools/fetch_sources.py --execute`、`tools/prepare.py`，再遵循上述官方数据说明准备资产。离线交付已经含固定源码，不需重新下载或重新安装依赖。

## 6. 正式训练与常见问题

完整原始 nuScenes 放在 `data/nuscenes_full/`，完整 OCC 展开在 `data/apollo/occ_gt_release_v1_0/`；官方两份 OCC train/val PKL 放在 `data/apollo/`。 官方 CAN bus 放在 `data/can_bus/`，至少需相应 scene-*_pose.json，空目录会被拒绝。

```bash
bash test/train_full_4p.sh --device_id=0,2,4,6 --data_path="$PWD/data/apollo"
```

完整入口检查数据版本并重新建立完整数据路径，缺少正式资产会退出。完整nuScenes需放入本模型数据目录，并按官方说明生成完整标注和CAN bus；mini标注不能复用为完整数据验收。 **本次没有执行完整数据训练，未验证完整训练收敛或精度。** `validate_1p.sh` 的当前判据对应本次功能配置，不能直接当成完整训练评估脚本；完整训练按上游官方评估说明验收。

| 提示 / 现象 | 处理 |
| --- | --- |
| 缺少镜像归档 / image ID 不一致 | 取得本模型完整包，核对 images.lock.json 与现有镜像用途 |
| 标注或图像路径缺失 | 重新运行本模型 prepare_data，核对 data 目录；不要生成假标签 |
| NPU 不空闲 / 卡号重复 | 核对宿主进程和物理映射，处理占用后再启动 |
| 只有训练日志没有通过报告 | 查看本次输出最后异常与退出码，不能据此宣称通过 |
| 迁入另一目录后文件路径错误 | 先 setup/prepare_data，再检查 plan，使用当前目录的新配置 |

本模型实机证据随验证报告提供；问题档案和内部操作记录留在本机，不随离线包上传。
