# 离线交付：先校验，再恢复

适用于不熟悉 Ascend 的使用者。完整交付包包含源码、五个实测镜像标签、公开 nuScenes mini、完整 Apollo OCC 归档、WorldEngine 功能数据、预训练权重、五模型保存结果和验证报告。下载整个文件夹，保留所有子目录和 `.partNNN` 分卷，不要逐个解压分卷。

## 1. 先校验下载是否完整

在交付根目录打开终端，有 Python 3.9+ 即可（Windows 也可校验）：

```bash
python restore_delivery.py verify --bundle .
```

最后应显示 `Delivery verification passed`。工具同时核对每卷大小、SHA256 和合并归档 SHA256；失败时根据提示补下对应分卷。`manifest.json` 固定源码提交、镜像 ID 和全部必须文件。额外报告见 `06_reports/`，整体文件清单见交付根目录 `SHA256SUMS.txt`。

## 2. 在 Ascend ARM64 Linux 机器恢复

机器需已具备配套 Ascend 驱动、Docker、Python 3.9+，至少 **300 GB 可用空间**（交付文件夹之外，用于合并缓存、数据展开和镜像加载），并检查宿主 `npu-smi info` 正常。驱动和固件属于宿主配套，不由交付包安装；软件栈与实测硬件见[环境指南](environment.md)。

将整份交付目录放到服务器，例如 `/var/ai-model/delivery/ascend-a3-5models-20261011`。执行：

```bash
cd /var/ai-model/delivery/ascend-a3-5models-20261011
python3 restore_delivery.py restore --bundle . --destination /var/ai-model/lzd-restored
```

**目标目录须为空或尚不存在。** 工具合并分卷、安全恢复文件，导入并核对镜像，解压 mini 和完整 OCC，重建数据绝对路径及 Apollo / WorldEngine 配置；准备容器关闭网络。恢复过程不访问上游下载服务。首次执行可能耗时数十分钟，看到 `Offline restore completed` 才算完成。

发生中断时保留交付文件、合并缓存和恢复目录，重跑相同命令。若提示目录不属于当前交付，应选择新的空目录；若镜像标签 ID 不同，应检查现有镜像用途后处理冲突。脚本不会改用未经验证的镜像 ID。

## 3. 选模型，运行功能检查

```bash
cd /var/ai-model/lzd-restored/migration
bash tools/quickstart.sh bevformer check
bash tools/quickstart.sh bevformer plan
```

同样替换模型名为 `sparse4d`、`qwen3vl8b`、`worldengine`、`apollo_visionnet`。`plan` 的 `missing_data: []` 表示 mini 所需核心文件已具备。确认所选卡空闲后，用 `mini` 做训练功能检查；Qwen / Apollo 用 `verify` 重载交付中的保存结果并检查推理。各模型命令、卡号和成功判据见[快速上手](quickstart.md)。

交付已包含功能验证所需数据，无需先执行 `download`。重新训练前按[保存旧结果](quickstart.md#重新训练前保存旧结果)改名保留整个对应输出目录，避免覆盖结果或触发 Qwen 自动续训。完整训练仍需正式数据和独立指标验收；mini 成功不表示完整数据精度、性能或长期稳定性通过。

## 文件分类

| 目录 / 文件 | 内容 |
| --- | --- |
| `01_code/` | 迁移仓源码、固定提交的上游源码与补丁 |
| `02_images/` | Docker 镜像归档分卷，含三个运行镜像与两个基础 / 回退镜像 |
| `03_data/` | 原始 mini / OCC 归档、标注、WorldEngine 官方功能资产 |
| `04_weights/` | 完整 Qwen3-VL-8B 权重与视觉模型初始化权重 |
| `05_results/` | 五模型验证产物和明确标注的历史对照产物 |
| `06_reports/` | 使用说明、版本核验、模型验证和离线恢复证据 |
| `manifest.json` / `restore_delivery.py` | 必须文件清单、版本锁定与恢复工具 |

Apollo OCC 完整原归档已校验；本次语义检查及训练 / 验证仍使用 mini 的 404 帧。WorldEngine 仅覆盖 AlgEngine 功能链路。来源许可证随源码保留，数据和权重使用范围遵循其原发布许可。
