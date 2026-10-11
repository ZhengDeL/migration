# Ascend 910C/A3 训练迁移

按 [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/README.CN.md) 的 `PyTorch/contrib/{cv,nlp}/<类别>/<模型>` 分类组织。每个模型有独立说明、补丁或配置和 `test/` 启动入口，上游提交及补丁哈希统一放在 [sources.lock.json](sources.lock.json)。

**第一次使用：打开[快速上手](docs/quickstart.md)，选择模型，按命令顺序运行。** 1# 已准备镜像和源码，使用 `bash tools/quickstart.sh <模型> <操作>` 即可，无需在宿主机安装 Python 包。

## 已在 1# 验证的内容

截至 2026-10-11，五个模型都在 1# 完成了短程数据训练。Apollo 和 WorldEngine 的真实样本四卡 10 步已通过公开 quickstart 入口核验；完整数据指标和长期稳定性仍待验收。详细范围及 checkpoint 哈希见[实测记录](docs/validation.md)。

| 模型入口名 | 分类与说明 | 物理卡数 | 当前实测结果 |
| --- | --- | ---: | --- |
| `bevformer` | [cv/detection/BEVFormer](PyTorch/contrib/cv/detection/BEVFormer/README.md) | 2 | 公开 nuScenes mini，1 epoch / 162 迭代，loss 有限、checkpoint 可加载 |
| `sparse4d` | [cv/detection/Sparse4D](PyTorch/contrib/cv/detection/Sparse4D/README.md) | 2 | 公开 nuScenes mini，10 迭代，loss 有限、checkpoint 可加载 |
| `qwen3vl8b` | [nlp/Qwen3-VL-8B](PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) | 2 | 演示图文数据 LoRA 10 步，参数更新、保存检查和保存后单图推理通过 |
| `worldengine` | [cv/others/WorldEngine](PyTorch/contrib/cv/others/WorldEngine/README.md) | 4 | AlgEngine：官方 NAVTEST 64 个时序锚点，官方基座初始化，四卡 10 步及保存状态检查通过 |
| `apollo_visionnet` | [cv/detection/Apollo-Vision-Net](PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) | 4 | 官方 162 帧 VAL / OCC 复用为功能训练，随机初始化，四卡 10 步及保存状态检查通过 |

1# 实测为 Atlas 800I A3 推理系列。当前实验成功不改变[官方训练硬件支持表](https://gitcode.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/docs/zh/install_guide.md)的支持范围；正式硬件验收需另有支持依据。2# 尚未接入，四卡结果均来自 1#。

## 实际用什么镜像

| 模型 | 1# 实际镜像 | 来源 |
| --- | --- | --- |
| BEVFormer、Sparse4D、Apollo | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` | 官方 DrivingSDK 26.1.0 A3 镜像，修正 headless OpenCV 导入依赖 |
| Qwen3-VL-8B | `local/llamafactory:a3-arm64-20261010` | LLaMA-Factory 提供的 Ascend A3 ARM64 镜像 |
| WorldEngine | `local/migration-worldengine:26.1.0-a3-20261011` | 上述 DrivingSDK 镜像派生，隔离安装 AlgEngine / nuPlan 地理依赖 |

完整官方标签、固定 ARM64 digest、实测 image ID、软件版本、Dockerfile 和新机器准备步骤都在[镜像与环境指南](docs/environment.md)。快速入口会核对实测 image ID；本地镜像不能直接通过 `docker pull local/...` 下载。

## 最短使用示例

```bash
cd /var/ai-model/lzd/migration
bash tools/quickstart.sh bevformer check
bash tools/quickstart.sh bevformer plan
bash tools/quickstart.sh bevformer mini
```

其他模型的完整命令见[快速上手](docs/quickstart.md)。五模型均可用 `plan` 查看 mini 命令、用 `mini` 做功能训练；WorldEngine / Apollo 另保留 `compute` 合成输入检查。完整数据准备齐全后才使用 `train`。默认宿主逻辑 ID 为两卡 `0,2`、四卡 `0,2,4,6`；在 1# 上它们分别属于不同物理卡，容器内重新编号为 `0,1` 或 `0,1,2,3`。运行前用宿主 `npu-smi info` 确认卡空闲。

## 从哪里继续

- [快速上手](docs/quickstart.md)：复制命令、看结果、处理常见提示。
- [镜像与环境](docs/environment.md)：镜像来源、固定版本、安装与数据准备。
- [社区调研](docs/community_research.md)：可复用的华为 / Ascend 实现、上游依据和缺口。
- [实测记录](docs/validation.md)：实际卡数、输入范围、检查结果和 checkpoint 哈希。

服务器可直接下载固定上游源码及公开权重。新机器应先准备镜像，再在相应容器的 Python 3.10+ 环境运行 `tools/fetch_sources.py` 和 `tools/prepare.py`；各模型说明给出补丁及数据要求。数据、模型权重、训练输出和登录信息不提交 GitHub。

## 验收范围

完整迁移验收还需真实训练数据、官方预训练初始化、代表性验证样本和对应指标：BEVFormer / Sparse4D 的 nuScenes mAP/NDS，Apollo 的检测 mAP / 占用 mIoU，WorldEngine AlgEngine 开环规划指标和单独适配的 SimEngine 闭环链路，Qwen 的独立业务图文评估。mini 与合成输入检查仅为功能证据。
