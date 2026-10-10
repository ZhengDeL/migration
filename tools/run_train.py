#!/usr/bin/env python3
"""Build and optionally run one pinned Ascend training command."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare import (LOCK, ROOT, compat_patch_for, patch_for, patch_state, require_checkout,
                     patch_bytes, verify_applied_patch, verify_sparse4d_files)


NPROC = {"bevformer": 2, "sparse4d": 2, "qwen3vl8b": 2,
         "worldengine": 4, "apollo_visionnet": 4}
DATA_FILES = {
    "bevformer": ["data/nuscenes/nuscenes_infos_temporal_train.pkl",
                  "data/nuscenes/nuscenes_infos_temporal_val.pkl",
                  "data/can_bus",
                  "ckpts/r101_dcn_fcos3d_pretrain.pth"],
    "sparse4d": ["data/nuscenes_anno_pkls/nuscenes_infos_train.pkl",
                 "data/nuscenes_anno_pkls/nuscenes_infos_val.pkl",
                 "data/nuscenes",
                 "nuscenes_kmeans900.npy", "ckpt/resnet50-19c8e357-modern.pth"],
    "apollo_visionnet": ["data/occ_gt_release_v1_0/nuscenes_infos_temporal_train_occ_gt.pkl",
                         "data/occ_gt_release_v1_0/nuscenes_infos_temporal_val_occ_gt.pkl",
                         "data/nuscenes",
                         "data/can_bus",
                         "ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth"],
    "worldengine": ["data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl",
                    "data/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl",
                    "data/raw/openscene-v1.1/sensor_blobs/trainval",
                    "data/alg_engine/test_8192_kmeans.npy",
                    "data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth"],
}


def sparse_config(source: Path, output: Path, smoke: bool,
                  dataset_version: str = "v1.0-trainval") -> Path:
    original = source / "projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py"
    content = original.read_text(encoding="utf-8")
    substitutions = {
        'dist_params = dict(backend="nccl")': 'dist_params = dict(backend="hccl")',
        "total_batch_size = 48": "total_batch_size = 2",
        "num_gpus = 8": "num_gpus = 2",
        "lr=6e-4,": "lr=2.5e-5,",
        'pretrained="ckpt/resnet50-19c8e357.pth"':
            'pretrained="ckpt/resnet50-19c8e357-modern.pth"',
    }
    for old, new in substitutions.items():
        if content.count(old) != 1:
            raise RuntimeError(f"Sparse4D config changed: expected one {old!r}")
        content = content.replace(old, new)
    if dataset_version == "v1.0-mini":
        old = 'version="v1.0-trainval"'
        if content.count(old) != 1:
            raise RuntimeError("Sparse4D config changed: expected one dataset version")
        content = content.replace(old, 'version="v1.0-mini"')
    if smoke:
        content += "\n# Only ten iterations for environment and backward smoke.\n"
        content += "runner = dict(type='IterBasedRunner', max_iters=10)\n"
        content += "checkpoint_config = dict(interval=100000)\n"
        content += "evaluation = dict(interval=100000, pipeline=vis_pipeline)\n"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "mini_smoke" if dataset_version == "v1.0-mini" else ("smoke" if smoke else "full")
    path = output / f"sparse4d_2p_{suffix}.py"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=NPROC)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--sdk", type=Path, help="required for BEVFormer and Sparse4D")
    parser.add_argument("--devices", required=True,
                        help="logical NPU indices inside this process/container; map host physical cards before container launch")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="ten optimizer steps or one epoch; no validation")
    parser.add_argument("--dataset-version", choices=("v1.0-trainval", "v1.0-mini"),
                        default="v1.0-trainval", help="nuScenes split; mini is for smoke only")
    parser.add_argument("--execute", action="store_true", help="run only after environment and data checks")
    parser.add_argument("--model-path", type=Path, help="local Qwen3-VL-8B checkpoint directory")
    parser.add_argument("--dataset-id", help="LLaMA-Factory dataset name from dataset_info.json")
    args = parser.parse_args()
    if args.dataset_version == "v1.0-mini" and (not args.smoke or args.model not in ("bevformer", "sparse4d")):
        parser.error("v1.0-mini is supported only for BEVFormer or Sparse4D --smoke")

    source = args.source.resolve(strict=True)
    sdk = args.sdk.resolve(strict=True) if args.sdk else None
    require_checkout(args.model, source)
    patch = patch_for(args.model, sdk)
    compat = compat_patch_for(args.model)
    if patch is not None and patch_state(source, patch) != "applied":
        raise RuntimeError("apply the reviewed patch with tools/prepare.py --apply first")
    if compat is not None and patch_state(source, compat) != "applied":
        raise RuntimeError("apply the compatibility patch with tools/prepare.py --apply first")
    if patch is not None:
        verify_applied_patch(source, patch, (compat,) if compat else ())
    if args.model == "sparse4d" and sdk is not None:
        verify_sparse4d_files(source, sdk)

    devices = args.devices.split(",")
    if len(devices) != NPROC[args.model] or len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        parser.error(f"{args.model} requires {NPROC[args.model]} distinct numeric device IDs")
    output = args.output_dir.resolve()
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")

    missing = [str(source / item) for item in DATA_FILES.get(args.model, ()) if not (source / item).exists()]
    if args.dataset_version == "v1.0-mini":
        mini_scene = source / "data/nuscenes/v1.0-mini/scene.json"
        if not mini_scene.is_file():
            missing.append(str(mini_scene))
    if args.model == "qwen3vl8b":
        if args.model_path is None or args.dataset_id is None:
            parser.error("Qwen3-VL requires --model-path and --dataset-id")
        model_dir = args.model_path.resolve()
        dataset_registry = source / "data/dataset_info.json"
        if not model_dir.is_dir():
            missing.append(str(model_dir))
        else:
            if not (model_dir / "config.json").is_file():
                missing.append(str(model_dir / "config.json"))
            revision_file = model_dir / ".source_revision"
            if not revision_file.is_file():
                missing.append(str(revision_file))
            elif revision_file.read_text(encoding="utf-8").strip() != LOCK["qwen3vl8b"]["checkpoint_revision"]:
                raise RuntimeError("Qwen model snapshot revision differs from sources.lock.json")
            for asset in ("tokenizer.json", "tokenizer_config.json",
                          "preprocessor_config.json", "video_preprocessor_config.json"):
                if not (model_dir / asset).is_file():
                    missing.append(str(model_dir / asset))
            weight_index = model_dir / "model.safetensors.index.json"
            if weight_index.is_file():
                index = json.loads(weight_index.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise RuntimeError("Qwen model weight index has no weight map")
                for shard in set(weight_map.values()):
                    if not (model_dir / shard).is_file() or (model_dir / shard).stat().st_size < 1024 * 1024:
                        missing.append(str(model_dir / shard))
            else:
                missing.append(str(weight_index))
        if not dataset_registry.is_file():
            missing.append(str(dataset_registry))
        else:
            registered = json.loads(dataset_registry.read_text(encoding="utf-8"))
            if args.dataset_id not in registered:
                missing.append(f"dataset-id:{args.dataset_id}")
            else:
                dataset_info = registered[args.dataset_id]
                file_name = dataset_info.get("file_name") if isinstance(dataset_info, dict) else None
                if file_name and not (source / "data" / file_name).is_file():
                    missing.append(str(source / "data" / file_name))
        if args.execute and not args.model_path.is_dir():
            raise RuntimeError(f"checkpoint directory missing: {args.model_path}")
        if args.execute and not (args.model_path / "config.json").is_file():
            raise RuntimeError("Qwen checkpoint config.json missing")
        if args.execute and not dataset_registry.is_file():
            raise RuntimeError("LLaMA-Factory data/dataset_info.json missing")
        if args.execute:
            model_config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
            if model_config.get("model_type") != "qwen3_vl":
                raise RuntimeError("checkpoint config.json is not a Qwen3-VL checkpoint")
            if args.dataset_id not in registered:
                raise RuntimeError(f"dataset {args.dataset_id!r} is not registered")
        if args.execute and not args.smoke and args.dataset_id == "mllm_demo":
            raise RuntimeError("demo dataset is not a full training dataset")
        env["FORCE_TORCHRUN"] = "1"
        env["NPROC_PER_NODE"] = str(NPROC[args.model])
        # The image supplies binary dependencies, while the pinned checkout
        # supplies LLaMA-Factory code. The CLI imports this src tree first.
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(source / "src"), env.get("PYTHONPATH", "")) if part)
        command = ["llamafactory-cli", "train", str(ROOT / "PyTorch/contrib/nlp/Qwen3-VL-8B/qwen3vl8b_lora_npu.yaml"),
                   f"model_name_or_path={args.model_path.resolve()}", f"dataset={args.dataset_id}",
                   f"output_dir={output}"]
        if args.smoke:
            command.append("max_steps=10")
        cwd = source
    elif args.model == "sparse4d":
        config = sparse_config(source, output, args.smoke, args.dataset_version)
        command = ["torchrun", "--standalone", "--nproc_per_node=2", "tools/train.py",
                   str(config), "--launcher", "pytorch", "--work-dir", str(output)]
        if args.smoke:
            command.append("--no-validate")
        cwd = source
    else:
        settings = {
            "bevformer": ("tools/fp16/train.py", "projects/configs/bevformer_fp16/bevformer_base_fp16.py", "5e-5", 24),
            "apollo_visionnet": ("tools/train.py", "projects/configs/bevformer/bev_tiny_det_occ_apollo.py", "1e-4", 24),
            "worldengine": ("scripts/train.py", "configs/worldengine/e2e_vadv2_50pct.py", "5e-5", 8),
        }
        entry, config, lr, epochs = settings[args.model]
        if args.model == "worldengine":
            cwd = source / "projects/AlgEngine"
            env["WORLDENGINE_ROOT"] = str(source)
        else:
            cwd = source
        epoch_count = 1 if args.smoke else epochs
        command = ["torchrun", "--standalone", f"--nproc_per_node={NPROC[args.model]}", entry,
                   config, "--launcher", "pytorch", "--work-dir", str(output)]
        if args.smoke:
            command.append("--no-validate")
        command += ["--cfg-options", "dist_params.backend=hccl", f"optimizer.lr={lr}",
                    "data.samples_per_gpu=1", f"runner.max_epochs={epoch_count}",
                    f"total_epochs={epoch_count}"]

    report = {"model": args.model, "revision": LOCK[args.model]["revision"],
              "patch_sha256": hashlib.sha256(patch_bytes(patch)).hexdigest() if patch else None,
              "compat_patch_sha256": hashlib.sha256(patch_bytes(compat)).hexdigest() if compat else None,
              "cwd": str(cwd), "devices": args.devices, "command": command,
              "dataset_version": args.dataset_version if args.model in ("bevformer", "sparse4d") else None,
              "missing_data": missing, "execute": args.execute}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if missing:
        raise RuntimeError("required data or weight files are missing")
    check = [sys.executable, str(ROOT / "tools/check_env.py"), "--model", args.model,
             "--nproc", str(NPROC[args.model]), "--output", str(output / "environment.json")]
    subprocess.run(check, env=env, check=True)
    if args.model in ("bevformer", "worldengine", "apollo_visionnet"):
        subprocess.run([sys.executable, str(ROOT / "tools/check_msda.py"),
                        "--output", str(output / "msda.json")], env=env, check=True)
    output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(f"run failed: {exc}")
