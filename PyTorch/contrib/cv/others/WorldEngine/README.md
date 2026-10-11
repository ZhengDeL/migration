# WorldEngine AlgEngine：Ascend 工程使用指南

文档复核：2026-10-12；实机验证：2026-10-11。覆盖 AlgEngine 真实图像短程训练与保存检查，当前使用 NAVTEST 功能子集。

## 1. 模型与环境

| 项目 | 本仓采用值 |
| --- | --- |
| 官方实现 | [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine/tree/3eded7e8ac64e5bed817860765089b5914ab698f)，固定提交 `3eded7e8ac64e5bed817860765089b5914ab698f` |
| 官方说明 | [数据组织](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/data_organization.md)、[AlgEngine 使用](https://github.com/OpenDriveLab/WorldEngine/blob/3eded7e8ac64e5bed817860765089b5914ab698f/docs/algengine_usage.md) |
| 配置 | `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py`，NAVFormer / TrajScoringHead |
| 资源 | ARM64 Linux，4 张空闲物理卡；1# 默认宿主逻辑 ID `0,2,4,6` |
| 镜像 | `local/migration-worldengine:26.1.0-a3-20261011` |
| 软件栈 | Python 3.10.21 / CANN 9.1.0 / Torch 2.7.1+cpu / TorchNPU 2.7.1.post6 / mx_driving 26.1.0.post2 |
| 额外依赖 | 隔离安装的 AlgEngine / nuPlan 依赖、NumPy 1.23.5、rasterio 1.3.11、Fiona 1.9.6 |

image ID 为 `sha256:ff85edc4a3e51385d814d9b3bfa442efdf040fc6c78e74e667ba9850f7b1b4a3`。镜像构建、OpenMMLab 版本和硬件支持边界见[环境指南](../../../../../docs/environment.md)。

本次覆盖 AlgEngine imitation；**SimEngine、3DGS、强化学习与完整后训练未适配**。NAVTEST 用于功能训练，不作为独立精度评估；配置名称中的 50% 不代表此次完成了正式 50% 训练。

## 2. 已有交付资产：检查并运行

新机器先完成[离线恢复](../../../../../docs/offline_delivery.md)。以下均从迁移仓根目录执行，恢复后替换示例 `cd` 路径。重新训练前按[保存旧结果](../../../../../docs/quickstart.md#重新训练前保存旧结果)保留 `outputs/worldengine/`。

```bash
cd /var/ai-model/lzd/migration
npu-smi info
bash tools/quickstart.sh worldengine check
bash tools/quickstart.sh worldengine plan
bash tools/quickstart.sh worldengine mini
```

`plan` 须显示 `missing_data: []`。mini 使用四卡 FP32、每卡 batch 1、AdamW 学习率 `5e-5`、10 步；保留四帧 / 八相机、正式图像流水线、200×200 BEV、六层 encoder 和完整 8192×40×3 词表。换卡示例：`ASCEND_HOST_DEVICES=8,10,12,14 bash tools/quickstart.sh worldengine mini`。

## 3. 从零准备源码和功能数据

先按环境指南准备专用镜像；在迁移仓根目录执行：

```bash
python3 tools/fetch_sources.py worldengine --execute
python3 tools/prepare.py worldengine --source sources/WorldEngine --apply
bash tools/quickstart.sh worldengine download
bash tools/quickstart.sh worldengine plan
```

download 获取约 **5.59 GB** 官方 NAVTEST 标注、PDM 缓存、感知基座 checkpoint 和 camera30 图像归档，另取约 4 MB 轨迹词表。发布快照、文件大小和 SHA 固定，支持续传；自动准备三段日志的 **140 帧 / 64 个时序锚点**。建议至少预留 20 GB 功能数据空间。

已有下载文件但缺少 prepared 文件时运行 `bash tools/quickstart.sh worldengine prepare`。必要布局在仓库父目录：

```text
data/worldengine_smoke/
├── nuplan_openscene_navtest.pkl
├── pdm_8192_gt_cache_navtest.pkl
├── track_map_nuplan_r50_navtrain_50pct_bs1x8.pth
├── openscene_sensor_test_camera_30.tgz
└── prepared/
    ├── manifest.json
    ├── worldengine_navtest_smoke.pkl
    ├── worldengine_navtest_smoke_pdm.pkl
    └── worldengine_navtest_smoke.yaml
```

词表位于 `sources/WorldEngine/data/alg_engine/test_8192_kmeans.npy`。数据准备选择真实连续帧并重写图像路径，保留坐标、轨迹标签和 PDM 数据，不启用改变数据语义的 diffusiondrive 模式。固定下载清单见[world_mini_data.py](../../../../../tools/world_mini_data.py)，子集条件见[数据 manifest](../../../../../verification/host1_20261011/worldengine_real/data_manifest.json)。

## 4. 成功判据与输出

| 检查 | 通过条件 / 结果位置 |
| --- | --- |
| 训练 | 命令退出码 0；`outputs/worldengine/mini/training_check_rank{0,1,2,3}.json` 全部为 `pass_training_smoke` |
| 计算内容 | 每 rank 十步原始输出、掩码前 imitation loss、259 个梯度张量有限，参数更新及被检查参数跨 rank 一致 |
| 保存 | rank 0 的 `checkpoint_reload` 为 `pass`，`optimizer_step_values` 为 `[10.0]` |
| checkpoint | `outputs/worldengine/mini/iter_10.pth`；本次 889 模型张量 / 259 优化器状态有限 |

字段结构和文件 SHA 见[验证记录](../../../../../docs/validation.md)。SDK `meta.iter=9` 是零基计数，优化器实际 step=10；保存检查未包含 checkpoint 续训。模型没有 quickstart verify 操作。`build` 仅构建模型，`compute` 是低分辨率合成输入检查，不能替代 mini 真实样本训练。

## 5. 正式训练与常见问题

正式训练按官方数据指南准备完整 OpenScene/NAVSIM 图像、navtrain/navtest 标注、两套 PDM 缓存、词表、基座 checkpoint 和场景划分；须符合上游 `data/raw/`、`data/alg_engine/` 与配置内路径。然后运行 `bash tools/quickstart.sh worldengine train`。四卡每卡 batch 1、学习率 `5e-5`、8 epoch 是实验起点，需独立规划指标、性能和稳定性验收。

| 提示 / 现象 | 处理 |
| --- | --- |
| prepared 文件缺失 | 完成 download；已有全部官方文件时执行 prepare |
| 词表缺失或哈希不匹配 | 运行固定下载入口核验，不能缩小或随机替代词表 |
| 地理依赖 / Matplotlib 导入失败 | 使用专用 WorldEngine 镜像并核对 ID，不直接混装到其他模型环境 |
| 四 rank 报告缺失 / 非有限 / 未更新 | 保留对应日志查错，不能只看 rank 0 或 checkpoint 存在 |

适配包括 NPU 分布式入口与变形注意力；兼容补丁修正 Matplotlib 样式名。版本及补丁哈希见[sources.lock.json](../../../../../sources.lock.json)。
