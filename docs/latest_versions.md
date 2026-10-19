# 固定版本与核验日期

下表记录本仓固定的复现基线及早期来源核验。[2026-10-19 实时来源复核](source_audit_20261019.json)确认八项当前范围：七项 GitHub 默认 HEAD 与固定提交相同；MobileNetV2 所在 GitCode 主仓仍为 2026-10-18 已核验的同一 HEAD，保留该提交对应的相同模型子树证据。没有自动升级源码或依赖。

2026-10-19 另执行了 LSTM、STGCN、MobileNetV2 的新全量 NPU 复验，见[实际数据](validation.md)。本轮没有重新执行这些模型的完整训练；来源核验、训练、复验与文档整理分别记录。WorldEngine AlgEngine 的后续版本和结果以[独立迁移仓](https://github.com/ZhengDeL/WorldEngine/tree/ascend-910c-algengine)为准。

| 模型/训练框架 | 已验证提交 | 来源核验日期 | 核验范围 | 证据 |
| --- | --- | --- | --- | --- |
| BEVFormer | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | 2026-10-12 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/detection/BEVFormer/validation/source_head.json) |
| Sparse4D | `249ffbb695f4e9db628d953e2bf6d36de04bbb69` | 2026-10-12 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/detection/Sparse4D/validation/source_head.json) |
| Qwen3-VL-8B | `1fded918ac4723ac9966363f51ed4500866c5d1d` | 2026-10-12 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/nlp/Qwen3-VL-8B/validation/source_head.json) |
| WorldEngine | `3eded7e8ac64e5bed817860765089b5914ab698f` | 2026-10-12 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/others/WorldEngine/validation/source_head.json) |
| Apollo-Vision-Net | `4909aa681bc435a74e793e714b8066d9bbf0095d` | 2026-10-12 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/detection/Apollo-Vision-Net/validation/source_head.json) |
| LSTM-Argoverse | `27696a19b3753a4ad8dcae3ad9c6107c5e4e237c` | 2026-10-18 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/others/LSTM-Argoverse/validation/source_audit_20261018.json) |
| STGCN-Traffic | `5cfa82c78984fc58fafe4841c35618c6647c4219` | 2026-10-18 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/others/STGCN-Traffic/validation/source_audit_20261018.json) |
| ResWorld | `a5ae973001dd6001d95b3389020a9e263d73a56e` | 2026-10-18 | 与当时默认分支 HEAD 一致 | [来源核验](../PyTorch/contrib/cv/others/ResWorld/validation/source_audit_20261018.json) |
| MobileNetV2 | `e779117e8cfeba296a7105e26adac0741ab04768` | 2026-10-18 | 当时主仓 HEAD 不同，但模型子树完全相同 | [来源核验](../PyTorch/contrib/cv/classification/MobileNetV2/validation/source_audit_20261018.json) |

Qwen 一行是 LLaMA-Factory 训练框架提交；官方 8B 权重固定为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，四分片 SHA256 见 [权重来源锁](../PyTorch/contrib/nlp/Qwen3-VL-8B/sources.lock.json)。WorldEngine 官方功能数据固定为 `d0e88d858533ca407a3387a26577c083aa7b3617`，见 [数据来源锁](../PyTorch/contrib/cv/others/WorldEngine/sources.lock.json)。

BEVFormer / Sparse4D 使用配套 DrivingSDK `branch_v26.1.0` 固定提交 `0a34fc1fbdf5c60015b23303bd01cb3e1a7fd837`；上游模型 HEAD 与适配 SDK 是不同的版本对象。MobileNetV2 子树为 `75fd4c46d94f560f776c57f48a53dc705c682de3`，不能仅凭仓库总体 HEAD 变化判定本模型代码改变。

## 运行环境

九模型的固定 tag、Docker image ID、软件组合和镜像归档 SHA256 见[环境与镜像](environment.md)。Qwen 当前独立工程使用 `local/llamafactory:a3-arm64-20261012` / `sha256:99d33965ae4502b07bffe2c3248887795ae0672a0e79dd0aaa567d09cce811b5`，其独立入口已于 2026-10-12 重跑。

后续升级应先比较本模型固定源码、补丁、环境和资产，再实际执行该模型的训练/验证及数值检查；通过后更新本模型锁文件与证据。来源核验日期、训练日期及文档更新时间分别记录。
