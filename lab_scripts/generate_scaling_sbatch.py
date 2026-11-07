#!/usr/bin/env python3
"""
Generate REVE production sbatch files for the 25%/50% scaling study,
using the best hyperparameters found by reve_optuna_search.py.

Usage:
    python3 generate_scaling_sbatch.py <fraction>   # fraction: 25 or 50

Reads:  /lustre10/scratch/nourmaj/reve_optuna_results/{dataset}_{arm}_{fraction}pct_best.json
Writes: /lustre10/scratch/nourmaj/reve_{dataset}_{arm}_{fraction}pct.sbatch
"""
import json
import sys
from pathlib import Path

SCRATCH = "/lustre10/scratch/nourmaj"
RESULTS_DIR = f"{SCRATCH}/reve_optuna_results"
N_EPOCHS = 50  # uniform across all 7 datasets in this study, per N_EPOCHS_OVERRIDE + task yaml defaults

# dataset key -> processed data folder root (matches reve_optuna_search.py's DATASET_CONFIG)
DATASET_ROOT = {
    "chbmit": "chb-mit-processed-80h",
    "clinical": "ds004504_processed",
    "dream": "dream_processed",
    "karaone": "karaone_processed",
    "yang2025": "yang2025_processed",
    "japaneeg": "japaneeg_processed_80h",
    "thoughtviz": "thoughtviz_processed",
}

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=reve_{dataset}_{arm}_{fraction}pct
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output={scratch}/reve_{dataset}_{arm}_{fraction}pct_%j.out
#SBATCH --error={scratch}/reve_{dataset}_{arm}_{fraction}pct_%j.err

source {scratch}/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE={scratch}/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd {scratch}/REVE/src
mkdir -p {scratch}/reve_outputs/{dataset}_{arm}_{fraction}pct
ln -sfn {scratch}/hf_cache {scratch}/reve_outputs/{dataset}_{arm}_{fraction}pct/.cache

# Best hyperparameters from reve_optuna_search.py (see
# {dataset}_{arm}_{fraction}pct_best.json): val_balanced_acc={best_val:.4f} over {n_trials} trials.
python dt.py \\
  task={dataset} \\
  training_mode=ft \\
  pretrained_path={pretrained_path} \\
  \\
  task.data_loader.dataset.path={scratch}/{data_root}_{fraction}pct \\
  task.fine_tuning.optimizer.lr={lr} \\
  task.fine_tuning.optimizer.weight_decay={weight_decay} \\
  task.fine_tuning.warmup_epochs={warmup_epochs} \\
  task.fine_tuning.n_epochs={n_epochs} \\
  trainer.clip_grad={clip_grad} \\
  seed=42 \\
  hydra.run.dir={scratch}/reve_outputs/{dataset}_{arm}_{fraction}pct \\
  cache_dir={scratch}/hf_cache
"""


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("25", "50"):
        print("Usage: python3 generate_scaling_sbatch.py <25|50>")
        sys.exit(1)
    fraction = sys.argv[1]

    written = []
    missing = []
    for dataset, data_root in DATASET_ROOT.items():
        for arm in ("pretrained", "random"):
            json_path = Path(RESULTS_DIR) / f"{dataset}_{arm}_{fraction}pct_best.json"
            if not json_path.exists():
                missing.append(str(json_path))
                continue

            with open(json_path) as f:
                result = json.load(f)

            params = result["best_params"]
            lr = params["lr"]
            grad_clip = params["grad_clip"]
            warmup_ratio = params["warmup_ratio"]
            weight_decay = result.get("weight_decay", 1e-5)
            best_val = result["best_val_balanced_acc"]
            n_trials = result["n_trials_run"]

            warmup_epochs = round(warmup_ratio * N_EPOCHS)
            clip_grad_str = "null" if grad_clip is None else str(grad_clip)
            pretrained_path = "hf:brain-bzh/reve-base" if arm == "pretrained" else '""'

            content = SBATCH_TEMPLATE.format(
                dataset=dataset,
                arm=arm,
                fraction=fraction,
                scratch=SCRATCH,
                data_root=data_root,
                lr=lr,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                n_epochs=N_EPOCHS,
                clip_grad=clip_grad_str,
                pretrained_path=pretrained_path,
                best_val=best_val,
                n_trials=n_trials,
            )

            out_path = Path(SCRATCH) / f"reve_{dataset}_{arm}_{fraction}pct.sbatch"
            with open(out_path, "w") as f:
                f.write(content)
            written.append(str(out_path))

    print(f"Written {len(written)} sbatch files:")
    for p in written:
        print(f"  {p}")
    if missing:
        print(f"\nMISSING {len(missing)} result json files (skipped):")
        for p in missing:
            print(f"  {p}")


if __name__ == "__main__":
    main()