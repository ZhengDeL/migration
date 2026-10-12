# LSTM 智驾研究证据

当前候选为 Argoverse 官方基础 LSTM 车辆轨迹预测网络，固定提交 `27696a19b3753a4ad8dcae3ad9c6107c5e4e237c`。

## 已验证的范围

2026-10-12 在 1# Atlas 800I A3 推理系列机器，使用一个物理卡的一个逻辑设备完成了原始编码器、解码器的 FP32 功能检查。本目录复验脚本的实测结果见 [neural_core_result.json](neural_core_result.json)。

| 项目 | 结果 |
| --- | --- |
| 输入 / 输出 | 合成坐标 `[2,20,2]` / `[2,30,2]` |
| 前向 CPU / NPU 最大绝对误差 | `4.470348358154297e-08` |
| 比较阈值 | rtol=0.001，atol=0.001 |
| 梯度与更新 | 14 个有限 NPU 梯度张量；一次 Adam 更新后 14 个参数张量改变 |
| 真实数据、双卡、完整训练及轨迹精度 | 未验证 |

基础 PyTorch 版本的 `+cpu` 后缀对应所装 wheel。脚本明确检查预测、loss、参数及梯度位于 NPU，并与相同初始权重的 CPU 前向输出比较。

## 镜像

| 项目 | 固定值 |
| --- | --- |
| tag | `local/migration-drivingsdk:26.1.0-a3-headless-20261010` |
| image ID | `sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2` |
| 架构 | ARM64 |
| CANN / PyTorch / TorchNPU | 9.1.0 / 2.7.1+cpu / 2.7.1.post6 |

详细版本及官方基础镜像 manifest 见 [images.lock.json](images.lock.json)。这是本次已实测的本地镜像；复验需要服务器已经导入该镜像。Git 中不包含镜像大文件，本目录尚不是完整模型离线交付包。

## 复验

在 ARM64 Linux Ascend 机器进入本目录，确保逻辑设备 `0` 空闲。以下命令与本轮实测采用相同的设备、内存上限和镜像；挂载范围为本目录及只读驱动。

```bash
image_tag='local/migration-drivingsdk:26.1.0-a3-headless-20261010'
image_id='sha256:34a01a4e2e46d6e2837af958ed91227411ea6eb4e1af29211c1494a3f0db89a2'
actual_image_id=$(docker image inspect --format '{{.Id}}' "$image_tag")
test "$actual_image_id" = "$image_id" || exit 1

docker run --rm --network none --memory 48g --cpus 4 \
  -v "$PWD":/work:rw -w /work \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0 -e OMP_NUM_THREADS=2 \
  -e MIGRATION_IMAGE_ID="$image_id" \
  --entrypoint /opt/conda/envs/bevformer/bin/python \
  "$image_tag" neural_core_probe.py
```

成功后生成 `runtime_results/neural_core_result.json`，`result` 为 `pass_original_neural_core_npu_feasibility`，并记录有限梯度、实际参数更新、CPU 对照误差及运行版本。任何断言或来源 hash 失败都以非零退出。仓库保存的历史结果不会被复验覆盖。

初次使用 8 GiB 容器主机内存上限时出现 cgroup OOM；调整为 48 GiB 后通过。这里的 `--memory` 是容器主机内存限制，不能据此推导模型的最小内存或 NPU HBM 需求。

## 来源与后续工作

[sources.lock.json](sources.lock.json)记录固定原文件的字节数、SHA256 和提交日期。复验脚本从原文件提取两个神经网络类，不执行旧版训练入口；来源许可见 [NOTICE.md](NOTICE.md)。

完整迁移还需验证真实 Argoverse 数据处理、坐标恢复、双卡 HCCL、checkpoint 保存重载、独立验证集 ADE / FDE / Miss Rate。选型说明见 [LSTM 智驾研究](../../docs/lstm_driving_research.md)。实验结果不改变 1# 的官方训练硬件支持范围。
