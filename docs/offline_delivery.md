# 独立模型离线交付

每个模型单独交付一个可直接查看的展开目录，以及一个对应的完整归档。模型目录中包含自己的指导、脚本、固定上游源码、运行镜像、功能数据、初始化权重和实测结果。镜像使用独立的 Docker save 归档，setup 导入后核对实际 image ID。

取用哪个模型，就按它的 README 执行。每份包的 `package_manifest.json` 和 `tools/verify_package.py` 校验本模型文件；归档另有 SHA256。目录名及上级路径可以改变，setup 会重建当前目录的数据路径。训练需要对应 Ascend ARM64 Linux 服务器；Windows 可查看文件并校验下载完整性。

| 单独交付目录 | 指导 |
| --- | --- |
| BEVFormer | [独立部署与训练](../PyTorch/contrib/cv/detection/BEVFormer/README.md) |
| Sparse4D | [独立部署与训练](../PyTorch/contrib/cv/detection/Sparse4D/README.md) |
| Qwen3-VL-8B | [独立部署与训练](../PyTorch/contrib/nlp/Qwen3-VL-8B/README.md) |
| WorldEngine | [独立部署与训练](../PyTorch/contrib/cv/others/WorldEngine/README.md) |
| Apollo-Vision-Net | [独立部署与训练](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/README.md) |

包内 `validation/` 保存本模型的验证依据和参考结果。setup 在本模型 `test/output/` 建立参考结果的工作副本；重新训练生成新的输出目录，并更新最近一次运行指针。原参考结果保留，最近一次失败也不会被旧结果掩盖。

公开交付仅包含运行和复验必要资产。内部问题台账、操作记录、失败尝试和服务排查日志保留在本机。原来的五模型合并分卷属于历史交付，当前独立目录不依赖它。

功能数据与正式训练数据的区别、卡数、检查内容和尚未验收的指标，均在每个模型自己的指导和验证报告中列明。完整数据精度、性能、长期稳定性以及正式训练硬件支持需另行验收。
