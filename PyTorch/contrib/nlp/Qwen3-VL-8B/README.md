# Qwen3-VL-8B-Instruct：Ascend LoRA 训练迁移

## 参考实现与范围

基座为 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，训练入口采用 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 的 `1fded918ac4723ac9966363f51ed4500866c5d1d`。本目录提供 BF16 LoRA SFT 配置：rank 8、每卡 batch 1、梯度累积 8、图像上限 262144 像素。使用官方 `qwen3_vl_nothink` 模板；具体业务模板需按数据目标确认。

## 环境和数据

按 LLaMA-Factory 的 [Ascend 环境说明](https://github.com/hiyouga/LlamaFactory/blob/main/docker/docker-npu/OVERVIEW.zh.md) 建立独立容器。训练前执行 `python3 tools/check_env.py --model qwen3vl8b --nproc 2`。准备完整的本地模型权重、processor/tokenizer 文件，在 LLaMA-Factory 的 `data/dataset_info.json` 注册图文训练集，并保留独立验证集。

## 快速开始

以下命令从仓库根目录执行；默认 `mllm_demo` 仅用于功能试跑，未传 `--execute` 时只打印计划：

```bash
python3 tools/fetch_sources.py qwen3vl8b --execute
python3 tools/fetch_qwen.py --execute
python3 tools/prepare.py qwen3vl8b --source sources/LLaMA-Factory
export QWEN_MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export QWEN_DATASET_ID=my_multimodal_dataset
bash PyTorch/contrib/nlp/Qwen3-VL-8B/test/train_2p.sh --smoke
bash PyTorch/contrib/nlp/Qwen3-VL-8B/test/train_2p.sh --smoke --execute
```

`--smoke` 跑 10 次优化迭代；完整训练移除该参数。脚本拒绝用 `mllm_demo` 做完整训练，并检查固定快照版本、模型类型、全部权重分片和数据集注册。两卡显存占用须在实机确认。

## 验证和状态

本地已验证 YAML 解析、上游模板名和启动命令；910C 训练尚未执行。实机需核对两 rank 有限 loss、LoRA 参数更新、checkpoint，以及独立图文样本的答案内容。仅有训练 loss 不代表多模态能力达标。
