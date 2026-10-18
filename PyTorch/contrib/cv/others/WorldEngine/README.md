# WorldEngine AlgEngine for PyTorch

本目录提供 Ascend 910C/A3 的迁移指导、训练及验证入口。完整离线 ZIP 包含本模型运行资产，解压后进入含本 README 的目录操作。

## 目录

- [概述](#概述)
- [准备训练环境](#准备训练环境)
- [开始训练](#开始训练)
- [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)

# 概述

## 简述

本工程迁移 WorldEngine 的 AlgEngine VADv2 imitation-learning 训练路径，使用真实时序图像与轨迹。

## 参考实现

- 参考实现：

  ```text
  url=https://github.com/OpenDriveLab/WorldEngine.git
  commit_id=3eded7e8ac64e5bed817860765089b5914ab698f
  ```

- 原仓训练／数据说明：[固定版本文档](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/algengine_usage.md)。
- 本页章节参考：[Ascend ModelZoo MobileNetV2 原指导](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/e779117e8cfeba296a7105e26adac0741ab04768/PyTorch/built-in/cv/classification/MobileNetV2_for_PyTorch/README.md)。
- 适配文件、补丁及来源摘要见本模型 [sources.lock.json](sources.lock.json)；镜像见 [images.lock.json](images.lock.json)。

本模型的 `test/`、`tools/` 是本项目独立入口；实际脚本名、卡数及实测配置在下文列出。原仓指标仍归原仓，本页结果均引用本项目实机记录。

- [上游数据／项目说明](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/data_organization.md)；[上游训练说明](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/algengine_usage.md)。

## 迁移说明

复用 DrivingSDK 变形注意力，按本目录 ascend.patch / compat.patch 处理 NPU 与依赖兼容；这里只覆盖 AlgEngine VADv2，SimEngine、3DGS、RL 未适配。

# 准备训练环境

## 安装昇腾环境

- 服务器：ARM64 Linux、配套 Ascend 驱动、Docker、Python 3.9+，训练需要 4 张空闲物理卡；模型 Python 与运行库在镜像内。`+cpu` 包版本标记不表示使用 CPU 训练，实际设备由环境检查和实测报告确认。
- 本次 1# 为 Atlas 800I A3 推理系列，实验训练结果不改变华为官方训练硬件支持范围。

宿主驱动与容器的 CANN/TorchNPU 配套需匹配。每张物理卡只取一个宿主逻辑 ID，具体拓扑先用 `npu-smi info` 核对。

## 安装模型环境

| 项目 | 本项目固定版本 |
| --- | --- |
| 镜像 tag | `local/migration-worldengine:26.1.0-a3-20261011` |
| image ID | `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3` |
| CANN | `9.1.0` |
| Python | `3.10.21` |
| PyTorch | `2.7.1+cpu` |
| TorchNPU | `2.7.1.post6` |
| DrivingSDK | `26.1.0.post2` |
| numpy | `1.23.5` |
| rasterio | `1.3.11` |
| fiona | `1.9.6` |

取得本模型完整离线包并解压，进入包含本 README 的目录。包内镜像为一个独立 Docker 归档；原始数据、源码和权重均属于本模型，数据准备不访问网络。

```bash
cd WorldEngine
python3 tools/verify_package.py
bash test/setup.sh
```

verify_package 显示 `Single-model package verification passed` 才表示下载文件完整；setup 导入/核对本模型镜像，核验源码和补丁，重建当前目录的数据路径。setup 不启动训练。若只有 GitHub 源码，需要先补齐本目录的数据和镜像资产，不能用空目录冒充离线包。

## 准备数据集和预训练权重

```text
data/worldengine_smoke/  # NAVTEST、PDM 缓存、camera30、基座 checkpoint
sources/WorldEngine/data/alg_engine/test_8192_kmeans.npy  # 完整轨迹词表
```

`sources.lock.json` 固定本模型源码和补丁。本模型的NPU与兼容补丁保存在本目录，来源和SHA见sources.lock.json。 迁移到新目录后重新运行本模型的 `bash test/prepare_data.sh`。

仅源码部署时，按上述原仓数据说明补齐资产；在本模型固定镜像中运行 `tools/fetch_sources.py --execute`、`tools/prepare.py`，再遵循上述官方数据说明准备资产。离线交付已经含固定源码，不需重新下载或重新安装依赖。

# 开始训练

## 训练模型与独立验证

```bash
npu-smi info
bash test/check_env.sh --device_id=0,2,4,6
bash test/plan.sh
bash test/train_4p.sh --device_id=0,2,4,6
bash test/validate_1p.sh --device_id=0
```

1# 一张物理卡有两个逻辑芯片；默认 `0,2,4,6` 每物理卡只取一个。容器内重新编号为连续 ID。换机先核对实际拓扑，换卡修改 `--device_id`。验证只用一张卡，只填一个 ID。

每次训练创建独立的 `test/output/functional_<时间>/`，保留已有结果；`test/output/latest_run.json` 指向最近一次尝试。训练失败时验证脚本拒绝把旧结果当成本次通过。所有原始终端输出、配置、checkpoint 和 JSON 报告均保存于本模型目录内。

本次功能配置：**NAVTEST 三段真实日志，140 帧 / 64 时序锚点；官方感知基座；4 卡 FP32 每卡 batch1，学习率5e-5，10 步**。输出 `iter_10.pth`。训练正常结束后自动检查保存内容，再运行 validate 做独立验证。

## 完整数据训练

完整数据保持官方布局放在 `data/worldengine_full/{raw,alg_engine}/`；保留源码中的 navtrain/navtest split 和完整词表。

```bash
bash test/train_full_4p.sh --device_id=0,2,4,6 --data_path="$PWD/data/worldengine_full"
```

完整入口检查数据版本并重新建立完整数据路径，缺少正式资产会退出。完整AlgEngine训练需官方navtrain/navtest图像、标注、两套PDM缓存、词表和基座；当前NAVTEST功能子集不能替代它。 **本次没有执行完整数据训练，未验证完整训练收敛或精度。** `validate_1p.sh` 的当前判据对应本次功能配置，不能直接当成完整训练评估脚本；完整训练按上游官方评估说明验收。

# 训练结果展示

## 本项目实测

独立训练／验证入口实机执行均退出 0；移动本模型目录后再次完成 setup、环境检查、数据计划及独立验证。依据：[入口实测](validation/entry_acceptance.json)、[换目录复验](validation/path_relocation.json)、[单镜像导入](validation/single_image_import.json)。

本页仅列本项目功能配置。完整数据的训练收敛、任务指标及性能以实际完成的独立验收为准。

## 通过判据

| 检查 | 判断方法 |
| --- | --- |
| 包完整性 | verify_package 退出0、文件大小与 SHA 匹配 |
| 环境 | `test/output/environment.json` 为 pass 或 pass_with_warnings，failures 为空；管理面告警须结合宿主 npu-smi 与实际计算判定 |
| 数据计划 | plan 的 missing_data 必须为 []；仅有文件还不代表训练通过 |
| 功能训练 | 命令退出0，当前运行目录 `train_result.json` 为 pass_functional_training；保存内容检查通过 |
| 独立验证 | validate 退出0，当前运行目录 `validation_result.json` 为 pass_independent_functional_validation |

验证内容：四 rank 原始输出、掩码前 imitation loss、梯度、参数更新与抽查同步；有限模型/优化器和 step10 保存重载。 每次新训练文件 SHA 可以不同，验收依据为对应本次运行的记录和检查内容。

仅 AlgEngine imitation 功能；NAVTEST 复用作训练，无独立 PDM 指标。SimEngine/3DGS/RL 尚未适配。 完整数据精度、性能、长期稳定性和 checkpoint 继续训练另行验收；不得把功能验证结果写成正式指标已通过。

# 版本说明

## 变更

- 2026-10-19：按固定版本 ModelZoo / DrivingSDK 原指导整理章节，补充原仓依据和本项目入口对应关系；训练代码、镜像、数据、已有数值门槛与实测记录保持。
- 此前训练与修复日期、实际输入和检查点身份见本模型 `validation/`；详细实验与复验命令保留在本模型指导中。

## FAQ

| 提示 / 现象 | 处理 |
| --- | --- |
| 缺少镜像归档 / image ID 不一致 | 取得本模型完整包，核对 images.lock.json 与现有镜像用途 |
| 标注或图像路径缺失 | 重新运行本模型 prepare_data，核对 data 目录；不要生成假标签 |
| NPU 不空闲 / 卡号重复 | 核对宿主进程和物理映射，处理占用后再启动 |
| 只有训练日志没有通过报告 | 查看本次输出最后异常与退出码，不能据此宣称通过 |
| 迁入另一目录后文件路径错误 | 先 setup/prepare_data，再检查 plan，使用当前目录的新配置 |

本模型实机证据随验证报告提供；问题档案和内部操作记录留在本机，不随离线包上传。
