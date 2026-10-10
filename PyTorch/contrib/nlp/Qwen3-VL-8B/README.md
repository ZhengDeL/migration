# Qwen3-VL-8B-Instruct：Ascend LoRA 训练迁移

## 参考实现与范围

基座为 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，训练入口采用 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 的 `1fded918ac4723ac9966363f51ed4500866c5d1d`。本目录提供 BF16 LoRA SFT 配置：rank 8、每卡 batch 1、梯度累积 8、图像上限 262144 像素。使用官方 `qwen3_vl_nothink` 模板；具体业务模板需按数据目标确认。

## 环境和数据

按本仓锁定提交的 LLaMA-Factory [Ascend 环境说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md) 建立独立容器。1# 实际使用 `local/llamafactory:a3-arm64-20261010`（Docker image ID `sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8`），来源为 `quay.io/ascend/llamafactory:latest-a3-openeuler` 的 ARM64 manifest `sha256:a95eeae05d63e2ec046c946a35301326aec742d914aed4bbe05caa572d478d36`。实测 CANN 9.1.0、Python 3.12.13、PyTorch 2.10.0+cpu、TorchNPU 2.10.0.post2、transformers 5.8.0、LLaMA-Factory 0.9.6.dev0；版本和归档 SHA256 详见[环境指南](../../../../docs/environment.md)。`run_train.py` 会优先从锁定的 LLaMA-Factory 源码 `src` 导入训练代码。训练前执行 `python3 tools/check_env.py --model qwen3vl8b --nproc 2`。准备完整的本地模型权重、processor/tokenizer 文件，在 LLaMA-Factory 的 `data/dataset_info.json` 注册图文训练集，并保留独立验证集。当前已在该镜像通过双卡基础 NPU/HCCL 和 SFT 配置解析；Qwen 模型级训练尚未通过。

## 快速开始

1# 上已准备镜像和完整模型权重时，从仓库根目录依次执行 `bash tools/quickstart.sh qwen3vl8b check`、`bash tools/quickstart.sh qwen3vl8b plan`、`bash tools/quickstart.sh qwen3vl8b mini`。命令作用、模型目录覆盖方法和通过标准见[快速上手](../../../../docs/quickstart.md)。

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

`--smoke` 跑 10 次优化迭代；完整训练移除该参数。脚本拒绝用 `mllm_demo` 做完整训练，并检查固定快照版本、模型类型、全部权重分片和数据集注册。`fetch_qwen.py` 对四个分片逐一核对模型发布方在该固定快照的 SHA256。两卡显存占用须在实机确认。

## 验证和状态

本地已验证 YAML 解析、上游模板名和启动命令；1# 已下载锁定的 LLaMA-Factory 源码，核对独立 NPU 镜像内容并通过该镜像的双 rank 基础训练/HCCL。实机还确认从锁定源码导入 LLaMA-Factory，按镜像实际依赖成功解析本仓 SFT 配置：`qwen3_vl_nothink`、LoRA rank 8、BF16、每卡 batch 1。Qwen 模型级训练尚未执行。实机需核对两 rank 有限 loss、LoRA 参数更新、checkpoint，以及独立图文样本的答案内容。仅有训练 loss 不代表多模态能力达标。
