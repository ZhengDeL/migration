# Qwen3-VL-8B-Instruct：Ascend 工程使用指南

文档复核：2026-10-12；实机验证：2026-10-11。覆盖两卡 BF16 LoRA SFT、保存状态检查和保存后图文推理。

## 1. 模型与环境

| 项目 | 本仓采用值 |
| --- | --- |
| 官方权重 | [Qwen3-VL-8B-Instruct 固定快照](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) |
| 训练源码 | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory/tree/1fded918ac4723ac9966363f51ed4500866c5d1d)，`1fded918ac4723ac9966363f51ed4500866c5d1d` |
| 官方昇腾参考 | [NPU 镜像说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md) |
| 资源 | ARM64 Linux；训练 2 张空闲物理卡，推理检查 1 张 |
| 镜像 | `local/llamafactory:a3-arm64-20261011` |
| 软件栈 | Python 3.12.13 / CANN 9.1.0 / Torch 2.10.0+cpu / TorchNPU 2.10.0.post2 / transformers 5.8.0 |

image ID 为 `sha256:5134fb97fcfe4a9a83103fbe45c6f196b847e4cfac2f873a29fb4effab8e7645`，启动器核对，训练优先导入锁定源码的 src。镜像来源、digest 和硬件支持边界见[环境指南](../../../../docs/environment.md)。

## 2. 已有交付资产：检查并运行

新机器先完成[离线恢复](../../../../docs/offline_delivery.md)。以下均在迁移仓根目录执行，恢复后替换示例 `cd` 路径。

**已有交付 adapter 时，可先运行 verify。重新训练前必须按[保存旧结果](../../../../docs/quickstart.md#重新训练前保存旧结果)保留整个 outputs/qwen3vl8b/。** 官方框架会发现旧 checkpoint 并尝试续训，或拒绝非空输出；直接复用目录不能保证获得一次新的十步检查。

```bash
cd /var/ai-model/lzd/migration
npu-smi info
bash tools/quickstart.sh qwen3vl8b check
bash tools/quickstart.sh qwen3vl8b plan
bash tools/quickstart.sh qwen3vl8b mini
bash tools/quickstart.sh qwen3vl8b verify
```

训练默认宿主逻辑 ID `0,2`，verify 默认只用 `0`。换卡分别设置 `ASCEND_HOST_DEVICES=8,10` 和 `ASCEND_HOST_DEVICES=8`。计划须显示 `missing_data: []`。

没有权重时先执行 `bash tools/quickstart.sh qwen3vl8b download`。默认下载到仓库父目录的 `weights/qwen3-vl-8b-instruct/`，支持续传；使用[Qwen 官方 ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)提交 `5d854aab08710c16b980ec6d603d863b3821b915`，四分片逐一对照 HF 固定快照 SHA。两站提交号不同，分片内容已核验一致。离线恢复后无需下载。

## 3. 源码、模型文件和业务数据

从零准备源码，在迁移仓根目录：

```bash
python3 tools/fetch_sources.py qwen3vl8b --execute
python3 tools/prepare.py qwen3vl8b --source sources/LLaMA-Factory
```

模型目录需要四个 safetensors 分片、索引、config、tokenizer、processor 文件和下载校验标记，只下载分片不能进行图文训练。自定义 `QWEN_MODEL_DIR` 必须位于 `MIGRATION_MOUNT_ROOT` 挂载目录内。文件 SHA 与版本见[sources.lock.json](../../../../sources.lock.json)。

业务图文数据按[官方数据格式](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/data/README_zh.md)准备。例如将以下 JSON 保存为 `sources/LLaMA-Factory/data/my_vl.json`，图片放在同一 data 目录下的 images/：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image>图中的交通标志是什么？"},
      {"from": "gpt", "value": "这是停车让行标志。"}
    ],
    "images": ["images/stop_sign.jpg"]
  }
]
```

在现有 `sources/LLaMA-Factory/data/dataset_info.json` 的顶层对象中**增加**条目，保留已有注册项：

```json
"my_vl": {
  "file_name": "my_vl.json",
  "formatting": "sharegpt",
  "columns": {"messages": "conversations", "images": "images"}
}
```

第二段是对象内的条目，不是完整 JSON 文件。每条样本的 `<image>` 数量与图片数一致，图片须真实存在且在容器中可读；标签按实际样本填写。另备独立业务验证集。

## 4. 配置、成功判据与输出

配置见[qwen3vl8b_lora_npu.yaml](qwen3vl8b_lora_npu.yaml)：`qwen3_vl_nothink` 模板、LoRA rank 8、目标 all、BF16、每卡 batch 1、图像上限 262144 像素、cutoff 2048、学习率 `1e-4`。正式配置累积 8、3 epoch；mini 临时改为累积 1、10 次更新、每步日志、第十步保存，使用上游 6 条 mllm_demo。

| 操作 | 通过条件 / 文件 |
| --- | --- |
| mini | 退出码 0，`outputs/qwen3vl8b/checkpoint_check.json` 为 `pass_functional` |
| 保存检查 | 十步 loss / 梯度有限、LoRA 更新、优化器 step=10、两个 rank 的 RNG 文件齐全 |
| verify | `outputs/qwen3vl8b/inference_check.json` 为 `pass_functional`，正确描述独立红色正方形 |

本次 504 个 adapter 张量有限、252 个 LoRA B 非零，文件 SHA 见[验证记录](../../../../docs/validation.md)。verify 只验一张合成图，不代表业务精度通过。

## 5. 正式训练与常见问题

注册业务数据并保留旧输出后：

```bash
QWEN_DATASET_ID=my_vl bash tools/quickstart.sh qwen3vl8b train
```

启动器拒绝用 mllm_demo 做正式训练。按业务数据量与显存评估配置，另验独立业务精度、性能和稳定性。

| 提示 / 现象 | 处理 |
| --- | --- |
| `dataset-id:...` 缺失 | 检查注册名、JSON 合法性与 file_name 对应文件 |
| 权重 / processor / 校验标记缺失 | 用完整交付或重跑 download，不手写标记绕过校验 |
| 输出目录已存在 / 自动续训 | 改名保留整个输出目录，重新 check、plan、mini |
| 图片数与标记不一致 | 按官方格式修正样本并核对容器内图片路径 |
| verify 设备数错误 | 单卡操作仅设置一个宿主逻辑 ID |
