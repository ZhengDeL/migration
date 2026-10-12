# 来源与使用范围

`upstream/` 保存 Argoverse 官方预测基线的固定原文件。来源、提交及 SHA256 见 `sources.lock.json`；原许可证为 BSD-3-Clause-Clear，完整文本保留在 `upstream/LICENSE`。

`neural_core_probe.py` 仅从固定源文件提取 `EncoderRNN`、`DecoderRNN` 两个原始类，不导入旧版训练工程的其他依赖。NPU 的设备选择、数学 fixture、梯度检查和结果记录由本项目添加。网络类的运算逻辑没有改写。

这些文件用于研究可复现性。真实 Argoverse 数据、地图、预处理、完整训练、checkpoint、双卡通信及轨迹评测尚未在本目录交付或验收。
