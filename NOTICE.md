# Third-party sources

This project does not vendor upstream model code, datasets, or weights. It pins upstream revisions in `sources.lock.json` and applies small patches to those checkouts.

- `PyTorch/contrib/cv/others/WorldEngine/ascend.patch` modifies [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine), Apache-2.0.
- `PyTorch/contrib/cv/detection/Apollo-Vision-Net/ascend.patch` modifies [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net), Apache-2.0.
- BEVFormer and Sparse4D preparation uses patches from [Ascend/DrivingSDK](https://gitcode.com/Ascend/DrivingSDK), BSD-3-Clause for its code; the individual model and dependency repositories retain their own licenses.
- The Qwen training entry uses [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), Apache-2.0, and [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) under its published model license.

Review each upstream license and dataset terms before redistribution or commercial training.
