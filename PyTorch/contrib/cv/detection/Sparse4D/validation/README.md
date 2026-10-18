# 本模型独立实机验证

本模型使用自己的训练入口及独立验证入口，容器只挂载本模型目录。`entry_acceptance.json` 记录实际执行时间、当次宿主逻辑卡号、命令和退出码；两条命令都退出0，才记录为 passed。

- `single_image_import.json`：空 Docker 环境导入本模型的一个运行镜像，核对 image ID 和归档 SHA。
- `source_head.json`：上游最新提交与当前固定版本的核验记录。
- `reference_run/train_result.json`：本次功能训练结果。
- `reference_run/saved_state_check.json`：实际保存模型、优化器及更新次数检查。
- `reference_run/validation_result.json`：本模型独立验证范围与结论。
- 原始终端日志随对应离线交付保留；GitHub 保存本目录列出的精简结果 JSON。

完整离线目录含参考 checkpoint；GitHub 仅保存精简验证依据。setup 建立可写的参考工作副本，validate 针对该副本复验。各项检查内容和未覆盖指标以本模型 README 为准，不能把功能训练通过写成正式精度、性能或长期稳定性验收。
