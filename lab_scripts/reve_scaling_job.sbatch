#!/bin/bash
#SBATCH --account=def-amiilab
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# One shared REVE scaling-run job script for every dataset/fraction/seed/arm
# combination -- do NOT sbatch this directly. Use submit_reve_job.sh instead,
# which computes the right --job-name/--output/--time for the dataset+arm
# and calls `sbatch [those flags] reve_scaling_job.sbatch --dataset=... ...`
# (same reasoning as biot_scaling_job.sbatch: #SBATCH lines are read
# literally before this script's shell code runs, so they can't depend on
# which dataset/arm was requested).
#
# Unlike BIOT, REVE needed NO repo code changes -- it already takes a clean
# CLI override, task.data_loader.dataset.path=..., so pointing it at a
# scaled folder is just changing that one argument. This script builds
# that argument, plus every other flag, to EXACTLY match each dataset's
# already-working pretrained/random sbatch scripts (verified against those
# scripts directly, not guessed).
#
# ISRUC is NOT included -- its REVE path (isruc_processed_80h) is raw
# subject/.npy data, not the flat pkl-per-trial format the other 7 datasets
# use, so it needs its own subsetting approach. Pending.
#
# Usage (via the wrapper, not directly):
#   bash submit_reve_job.sh --dataset=chbmit --fraction=50 --seed=1 --arm=pretrained
#
# --dataset: one of chbmit, clinical, dream, karaone, thoughtviz, yang2025, japaneeg
# --fraction: 100, 50, or 25
# --seed: 1, 2, or 42

set -e

DATASET=""
FRACTION=""
SEED=""
ARM=""
for arg in "$@"; do
    case $arg in
        --dataset=*) DATASET="${arg#*=}" ;;
        --fraction=*) FRACTION="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        --arm=*) ARM="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
for v in DATASET FRACTION SEED ARM; do
    if [ -z "${!v}" ]; then
        echo "Missing required --${v,,}=... argument"; exit 1
    fi
done
if [ "$ARM" != "pretrained" ] && [ "$ARM" != "random" ]; then
    echo "--arm must be 'pretrained' or 'random', got: $ARM"; exit 1
fi
if [ "$FRACTION" != "100" ] && [ "$FRACTION" != "50" ] && [ "$FRACTION" != "25" ]; then
    echo "--fraction must be 100, 50, or 25, got: $FRACTION"; exit 1
fi

PRETRAINED_PATH="hf:brain-bzh/reve-base"

# Per-dataset config: TASK|ORIGINAL_PATH|extra flags per arm.
# EXTRA_PRETRAINED / EXTRA_RANDOM hold whatever extra hyperparameter
# overrides that dataset's existing scripts use (n_epochs, optimizer.lr) --
# copied EXACTLY from each dataset's own working pretrained/random scripts,
# nothing assumed or extrapolated across datasets.
case "$DATASET" in
    chbmit)
        TASK="chbmit"; ORIGINAL_PATH="/scratch/nourmaj/chb-mit-processed-80h"
        EXTRA_PRETRAINED="task.fine_tuning.n_epochs=25"
        EXTRA_RANDOM="task.fine_tuning.n_epochs=25 task.fine_tuning.optimizer.lr=0.00003"
        ;;
    clinical)
        TASK="clinical"; ORIGINAL_PATH="/scratch/nourmaj/ds004504_processed"
        EXTRA_PRETRAINED=""
        EXTRA_RANDOM="task.fine_tuning.optimizer.lr=0.00003"
        ;;
    dream)
        TASK="dream"; ORIGINAL_PATH="/scratch/nourmaj/dream_processed"
        EXTRA_PRETRAINED=""
        EXTRA_RANDOM="task.fine_tuning.optimizer.lr=0.00003"
        ;;
    karaone)
        TASK="karaone"; ORIGINAL_PATH="/scratch/nourmaj/karaone_processed"
        EXTRA_PRETRAINED=""
        EXTRA_RANDOM="task.fine_tuning.optimizer.lr=0.00003"
        ;;
    thoughtviz)
        TASK="thoughtviz"; ORIGINAL_PATH="/scratch/nourmaj/thoughtviz_processed"
        EXTRA_PRETRAINED=""
        EXTRA_RANDOM="task.fine_tuning.optimizer.lr=0.00003"
        ;;
    yang2025)
        TASK="yang2025"; ORIGINAL_PATH="/scratch/nourmaj/yang2025_processed"
        EXTRA_PRETRAINED="task.fine_tuning.n_epochs=25"
        EXTRA_RANDOM="task.fine_tuning.n_epochs=25 task.fine_tuning.optimizer.lr=0.00003"
        ;;
    japaneeg)
        TASK="japaneeg"; ORIGINAL_PATH="/scratch/nourmaj/scratch_iteration2/japaneeg_processed"
        EXTRA_PRETRAINED=""
        EXTRA_RANDOM="task.fine_tuning.optimizer.lr=0.00003"
        ;;
    isruc)
        echo "isruc is not configured for REVE scaling yet -- its data path is raw subject/.npy format, not the flat pkl folders the other datasets use. Needs its own subsetting approach first."
        exit 1
        ;;
    *)
        echo "Unknown --dataset=$DATASET"; exit 1
        ;;
esac

# ---- resolve data path for the requested fraction ----
if [ "$FRACTION" == "100" ]; then
    DATA_PATH="$ORIGINAL_PATH"
else
    DATA_PATH="${ORIGINAL_PATH}_${FRACTION}pct"
fi
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: expected data folder not found: $DATA_PATH"
    exit 1
fi

# ---- pretrained vs random ----
if [ "$ARM" == "pretrained" ]; then
    ARM_FLAGS="pretrained_path=$PRETRAINED_PATH $EXTRA_PRETRAINED"
else
    ARM_FLAGS="$EXTRA_RANDOM"
fi

RUN_DIR="/scratch/nourmaj/scratch_iteration2/reve_scaling_outputs/${DATASET}_${FRACTION}pct_s${SEED}_${ARM}"

echo "=== REVE scaling run ==="
echo "dataset=$DATASET fraction=$FRACTION seed=$SEED arm=$ARM"
echo "data_path=$DATA_PATH"
echo "run_dir=$RUN_DIR"
echo "========================"

source ~/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/nourmaj/REVE/src
mkdir -p "$RUN_DIR"
ln -sfn $SCRATCH/hf_cache "$RUN_DIR/.cache"

python dt.py task=$TASK training_mode=ft \
    task.data_loader.dataset.path=$DATA_PATH \
    $ARM_FLAGS \
    seed=$SEED \
    hydra.run.dir=$RUN_DIR \
    cache_dir=$SCRATCH/hf_cache