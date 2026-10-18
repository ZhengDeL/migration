# Third-party sources

This repository contains migration code and patches, together with selected pinned upstream files under each model’s `vendor/` directory. Larger upstream checkouts, datasets, model weights, runtime images and trained checkpoints are provided or prepared separately. Each model has its own `sources.lock.json`, runtime image lock and applicable source/asset notices.

- BEVFormer and Sparse4D use [Ascend DrivingSDK](https://gitcode.com/Ascend/DrivingSDK) examples and patches; retained SDK helper files and individual model repositories keep their original licenses.
- WorldEngine adapts [OpenDriveLab/WorldEngine](https://github.com/OpenDriveLab/WorldEngine), Apache-2.0.
- Apollo VisionNet adapts [ApolloAuto/Apollo-Vision-Net](https://github.com/ApolloAuto/Apollo-Vision-Net), Apache-2.0.
- Qwen training uses [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), Apache-2.0, and [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) under its published model license.
- LSTM-Argoverse retains [Argoverse forecasting](https://github.com/jagjeet-singh/argoverse-forecasting) baseline files, BSD-3-Clause-Clear; see the model notice and retained upstream license.
- STGCN-Traffic uses [hazdzz/STGCN](https://github.com/hazdzz/STGCN) and references [the original paper implementation](https://github.com/VeritasYin/STGCN_IJCAI-18); both applicable source licenses are retained in the model.
- ResWorld retains [mengtan00/ResWorld](https://github.com/mengtan00/ResWorld), Apache-2.0.
- MobileNetV2 retains the selected [Ascend ModelZoo-PyTorch](https://gitcode.com/Ascend/ModelZoo-PyTorch) model files and their license.

Review each retained source license, dataset terms and model license for the intended redistribution and use. Repository-level licensing does not replace third-party terms.
