# 1# 双卡快速上手

本页适用于已准备好模型源码、标注、权重和本仓镜像的 1#。先在服务器进入仓库目录。`check` 只检查环境，`plan` 列出缺少的文件，`mini` 才开始公开 nuScenes mini 或 Qwen 演示数据的功能训练；`train` 需要正式数据。三个模型分别运行，避免争用同两张卡。

```bash
cd /var/ai-model/lzd/migration
npu-smi info                         # 确认宿主物理卡 0、1 可用
bash tools/quickstart.sh bevformer check
bash tools/quickstart.sh bevformer plan
bash tools/quickstart.sh bevformer mini
```

把上面命令中的 `bevformer` 换成 `sparse4d` 即可检查或运行 Sparse4D。`qwen3vl8b` 也使用同一入口；先运行 `download` 从 Qwen 官方 ModelScope 仓下载固定提交并核对四个权重分片的 SHA256，然后运行 `plan` 查看是否还缺文件。默认模型目录为 `/var/ai-model/lzd/weights/qwen3-vl-8b-instruct`；可用 `QWEN_MODEL_DIR` 指定同一挂载目录内的其他路径。下载可续传，但必须看到校验完成及 `plan` 的 `missing_data` 为空后才执行 `mini`。1# 需要经现场批准的网络代理时，先为当前 shell 配置 `HTTPS_PROXY`、`HTTP_PROXY`；脚本只把它们传给下载容器。Qwen 的 `mini` 使用 LLaMA-Factory 自带 `mllm_demo`，正式训练必须在源码 `data/dataset_info.json` 注册自己的图文数据集，并设置 `QWEN_DATASET_ID`。

```bash
bash tools/quickstart.sh sparse4d check
bash tools/quickstart.sh sparse4d plan
bash tools/quickstart.sh sparse4d mini

bash tools/quickstart.sh qwen3vl8b check
bash tools/quickstart.sh qwen3vl8b download
bash tools/quickstart.sh qwen3vl8b plan
bash tools/quickstart.sh qwen3vl8b mini
```

| 模型 | 1# 使用的镜像 | `mini` 前还需要 |
| --- | --- | --- |
| BEVFormer | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` | nuScenes mini、CAN bus 目录、mini 标注 pkl、官方预训练权重 |
| Sparse4D | 同一镜像中的独立 `sparse4d` 环境 | nuScenes mini、mini 标注 pkl、mini 锚框、已转换的 ResNet50 权重 |
| Qwen3-VL-8B | `local/llamafactory:a3-arm64-20261010` | 完整且校验通过的模型快照、演示图文数据 |

`quickstart.sh` 会核对镜像 ID，挂载两张宿主物理卡及 Ascend 驱动，将它们映射为容器内 `0,1`，使用相应容器的 Python。默认宿主逻辑设备为 `0,2`，它们在 1# 上属于两张不同物理卡；改变选卡时先核对 `npu-smi info` 中的物理卡与芯片映射，再设置 `ASCEND_HOST_DEVICES=ID1,ID2`。若代码或数据不在同一个父目录下，设置 `MIGRATION_MOUNT_ROOT` 为同时包含仓库及数据的目录。脚本会把该目录以相同绝对路径挂载进容器。

**怎样判断通过：** `check` 的 JSON 报告为 `pass` 或已核对宿主 `npu-smi info` 的 `pass_with_warnings`；`plan` 的 `missing_data` 为空；`mini` 正常退出，两个 rank 日志均有有限 loss，并生成可加载 checkpoint。BEVFormer 和 Sparse4D 已在 1# 达到这一级功能验证。`train` 之前还要准备完整数据、预训练权重并确认业务指标；mini 结果不能代替正式精度和稳定性验收。

缺少文件时，先按[BEVFormer](../PyTorch/contrib/cv/detection/BEVFormer/README.md)、[Sparse4D](../PyTorch/contrib/cv/detection/Sparse4D/README.md)或[Qwen3-VL-8B](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md)的“环境和数据”准备，再重试 `plan`。镜像下载、固定 digest 和版本见[环境指南](environment.md)。WorldEngine、Apollo VisionNet 尚未选定可运行镜像或在 2# 实训，不使用本快速入口。
