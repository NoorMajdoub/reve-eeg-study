"""
Generate REVE production sbatch files from the completed Optuna search
results, for all 8 datasets x 2 arms (16 files total).

Why regenerate instead of sed-patching the existing 16 files in place:
each file needs several coordinated overrides added (lr, weight_decay,
clip_grad, warmup_epochs, explicit n_epochs=50), and a couple of the
existing files (e.g. reve_clinical_random.sbatch) already have a stale
placeholder lr override in them. Patching text in-place risks leaving a
duplicate/stale override key in the same `python dt.py ...` invocation,
which Hydra can error on (or worse, silently apply in an order you didn't
expect). Regenerating the whole invocation from one known-correct template
avoids that class of mistake entirely.

Safety: this writes into ~/reve_production_v2/ (NOT over your existing
reve_*.sbatch files), so you can diff and review before replacing anything.

What's applied per file, on top of what production already had:
  - task.fine_tuning.optimizer.lr        <- best_params.lr from the search
  - task.fine_tuning.optimizer.weight_decay <- 1e-5 (fixed, same as BIOT)
  - trainer.clip_grad                    <- best_params.grad_clip (or null)
  - task.fine_tuning.warmup_epochs       <- round(warmup_ratio * n_epochs)
  - task.fine_tuning.n_epochs=50         <- explicit on EVERY file now
    (not just japaneeg/yang2025, which defaulted to 25 -- making it
    explicit everywhere means no file's behavior depends on a yaml
    default silently matching what we want)
  - random arm: pretrained_path=""       <- explicit empty override,
    rather than relying on the pretrained checkpoint file simply not
    existing on disk (the same safety improvement flagged earlier and
    never yet applied to the production scripts -- doing it now while
    we're already touching every file)
  - --mem=64G already included in the sbatch header (the bug fixed
    earlier across all 16 existing production files)

Run this on Nibi: python3 generate_reve_production_sbatch.py
"""
import json
import math
import os

RESULTS_DIR = "/scratch/nourmaj/reve_optuna_results"
OUT_DIR = os.path.expanduser("~/reve_production_v2")
N_EPOCHS = 50  # uniform across every dataset for the final production runs

# Same roots reve_optuna_search.py used -- already confirmed matching each
# task yaml's data_loader.dataset.path.
DATASET_CONFIG = {
    "chbmit":     "/scratch/nourmaj/chb-mit-processed-80h",
    "clinical":   "/scratch/nourmaj/ds004504_processed",
    "dream":      "/scratch/nourmaj/dream_processed",
    "karaone":    "/scratch/nourmaj/karaone_processed",
    "yang2025":   "/scratch/nourmaj/yang2025_processed",
    "japaneeg":   "/scratch/nourmaj/japaneeg_processed_80h",
    "isruc":      "/scratch/nourmaj/isruc_processed_80h",
    "thoughtviz": "/scratch/nourmaj/thoughtviz_processed",
}

PRETRAINED_HF_ID = "hf:brain-bzh/reve-base"
FIXED_WEIGHT_DECAY = 1e-5

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=reve_{dataset}_{arm}
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/nourmaj/reve_{dataset}_{arm}_%j.out
#SBATCH --error=/scratch/nourmaj/reve_{dataset}_{arm}_%j.err

source ~/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nourmaj/REVE/src
mkdir -p $SCRATCH/reve_outputs/{dataset}_{arm}
ln -sfn $SCRATCH/hf_cache $SCRATCH/reve_outputs/{dataset}_{arm}/.cache

# Best hyperparameters from reve_optuna_search.py (see
# {results_file}): val_balanced_acc={best_val:.4f} over {n_trials} trials.
python dt.py \\
  task={dataset} \\
  training_mode=ft \\
  {pretrained_line}\\
  task.data_loader.dataset.path={data_root} \\
  task.fine_tuning.optimizer.lr={lr} \\
  task.fine_tuning.optimizer.weight_decay={weight_decay} \\
  task.fine_tuning.warmup_epochs={warmup_epochs} \\
  task.fine_tuning.n_epochs={n_epochs} \\
  trainer.clip_grad={clip_grad} \\
  seed=42 \\
  hydra.run.dir=$SCRATCH/reve_outputs/{dataset}_{arm} \\
  cache_dir=$SCRATCH/hf_cache
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'dataset':<12} {'arm':<11} {'lr':>12} {'clip':>6} {'warmup_ep':>9}  best_val_balanced_acc")

    missing = []
    for dataset in DATASET_CONFIG:
        for arm in ["pretrained", "random"]:
            results_file = f"{dataset}_{arm}_100pct_best.json"
            path = os.path.join(RESULTS_DIR, results_file)
            if not os.path.exists(path):
                missing.append(path)
                continue

            with open(path) as f:
                result = json.load(f)

            lr = result["best_params"]["lr"]
            grad_clip = result["best_params"]["grad_clip"]
            warmup_ratio = result["best_params"]["warmup_ratio"]
            warmup_epochs = round(warmup_ratio * N_EPOCHS)
            clip_grad_str = "null" if grad_clip is None else grad_clip

            pretrained_line = (
                f"pretrained_path={PRETRAINED_HF_ID} \\\n  "
                if arm == "pretrained"
                else 'pretrained_path="" \\\n  '
            )

            content = SBATCH_TEMPLATE.format(
                dataset=dataset,
                arm=arm,
                pretrained_line=pretrained_line,
                data_root=DATASET_CONFIG[dataset],
                lr=lr,
                weight_decay=FIXED_WEIGHT_DECAY,
                warmup_epochs=warmup_epochs,
                n_epochs=N_EPOCHS,
                clip_grad=clip_grad_str,
                results_file=results_file,
                best_val=result["best_val_balanced_acc"],
                n_trials=result["n_trials_run"],
            )

            out_path = os.path.join(OUT_DIR, f"reve_{dataset}_{arm}.sbatch")
            with open(out_path, "w") as f:
                f.write(content)

            print(f"{dataset:<12} {arm:<11} {lr:>12.4e} {str(grad_clip):>6} {warmup_epochs:>9}  {result['best_val_balanced_acc']:.4f}  -> {out_path}")

    if missing:
        print(f"\n!!! Missing {len(missing)} results file(s), skipped:")
        for m in missing:
            print(f"  - {m}")
    else:
        print(f"\nAll 16 sbatch files generated in {OUT_DIR}/")
        print("Review with: diff ~/reve_clinical_pretrained.sbatch ~/reve_production_v2/reve_clinical_pretrained.sbatch")
        print("Once satisfied, copy into place: cp ~/reve_production_v2/*.sbatch ~/")


if __name__ == "__main__":
    main()