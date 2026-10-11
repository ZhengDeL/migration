# Qwen3-VL-8B-Instruct：Ascend LoRA 训练迁移

## 参考实现与范围

基座为 [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)，训练入口采用 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) 的 `1fded918ac4723ac9966363f51ed4500866c5d1d`。本目录提供 BF16 LoRA SFT 配置：rank 8、每卡 batch 1、梯度累积 8、图像上限 262144 像素。使用官方 `qwen3_vl_nothink` 模板；具体业务模板需按数据目标确认。

## 环境和数据

按本仓锁定提交的 LLaMA-Factory [Ascend 环境说明](https://github.com/hiyouga/LlamaFactory/blob/1fded918ac4723ac9966363f51ed4500866c5d1d/docker/docker-npu/OVERVIEW.zh.md) 建立独立容器。1# 实际使用 `local/llamafactory:a3-arm64-20261010`（Docker image ID `sha256:8aac9350905d1aa8b6bdf2cffc6d512d64f7e936d6ed20cf7e4076726be372c8`），来源为 `quay.io/ascend/llamafactory:latest-a3-openeuler` 的 ARM64 manifest `sha256:a95eeae05d63e2ec046c946a35301326aec742d914aed4bbe05caa572d478d36`。实测 CANN 9.1.0、Python 3.12.13、PyTorch 2.10.0+cpu、TorchNPU 2.10.0.post2、transformers 5.8.0、LLaMA-Factory 0.9.6.dev0；版本和归档 SHA256 详见[环境指南](../../../../docs/environment.md)。`run_train.py` 会优先从锁定的 LLaMA-Factory 源码 `src` 导入训练代码。准备完整的本地模型权重、processor/tokenizer 文件，在 LLaMA-Factory 的 `data/dataset_info.json` 注册图文训练集，并保留独立验证集。该镜像已通过双卡基础 NPU/HCCL、十步 LoRA 功能训练和保存后的图文推理；业务数据训练与精度尚未验收。

## 快速开始

1# 上已准备镜像时，从仓库根目录依次执行 `bash tools/quickstart.sh qwen3vl8b check`、`bash tools/quickstart.sh qwen3vl8b download`、`bash tools/quickstart.sh qwen3vl8b plan`；确认 `missing_data` 为空后执行 `bash tools/quickstart.sh qwen3vl8b mini`。`download` 使用 [Qwen 官方 ModelScope 仓](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)的固定提交 `5d854aab08710c16b980ec6d603d863b3821b915`，逐分片对照 `sources.lock.json` 中 Hugging Face 固定快照的 SHA256。两仓的提交号不同，校验通过表示四个权重分片内容一致；各自来源标记分别为 `.source_revision` 和 `.source_modelscope_revision`。命令作用、代理配置、模型目录覆盖方法和通过标准见[快速上手](../../../../docs/quickstart.md)。

以下命令从仓库根目录执行；默认 `mllm_demo` 仅用于功能试跑，未传 `--execute` 时只打印计划：

```bash
python3 tools/fetch_sources.py qwen3vl8b --execute
python3 tools/fetch_qwen.py --execute
# 若 Hugging Face 线路不稳，可使用 Qwen 官方 ModelScope 仓：
python3 tools/fetch_qwen.py --hub modelscope --execute
python3 tools/prepare.py qwen3vl8b --source sources/LLaMA-Factory
export QWEN_MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export QWEN_DATASET_ID=my_multimodal_dataset
bash PyTorch/contrib/nlp/Qwen3-VL-8B/test/train_2p.sh --smoke
bash PyTorch/contrib/nlp/Qwen3-VL-8B/test/train_2p.sh --smoke --execute
```

`--smoke` 跑 10 次优化迭代，临时将梯度累积改为 1、逐步输出 loss，并在第 10 步保存；正式训练移除该参数，使用 YAML 中的梯度累积 8。脚本拒绝用 `mllm_demo` 做完整训练，并检查固定快照版本、模型类型、全部权重分片和数据集注册。`fetch_qwen.py` 对四个分片逐一核对模型发布方在该固定快照的 SHA256。smoke 正常结束后自动运行 `tools/check_qwen_checkpoint.py`，检查保存步数、逐步 loss/梯度、LoRA 更新、优化器和两个 rank 的保存文件；结果为 `pass_functional`，完整报告存入训练输出目录的 `checkpoint_check.json`。

## 验证和状态

2026-10-11，1# 使用两张不同物理卡完成 BF16 LoRA 十步训练，两个进程均报告 `world size: 2`、`device: npu:0/1`。演示集有 6 条记录、3 张图片；十步 loss 全部有限，首/末为 3.2745/2.5068，范围 1.9903–3.4865。checkpoint 含 504 个有限适配器张量，252 个 LoRA B 张量全部非零（默认初始化为零），504 个优化器状态的步数均为 10，并含两个 rank 的 RNG 文件。适配器文件为 87,368,144 字节，SHA256 `7ee34281193045e042789faa20a570fb6cf3ddc16df1c8aee0923c5640839c7e`。

另在单 NPU 重新加载保存后的适配器，对演示集之外的白底红色正方形图片提问，回答正确指出红色、正方形和白色背景。这只验证一张合成图的加载与图文执行，业务场景、多模态精度、性能和稳定性仍需独立验收。
