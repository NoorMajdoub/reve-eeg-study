#!/bin/bash
#SBATCH --job-name=reve_optuna
# (overridden per-submission by submit_reve_optuna_job.sh's --job-name flag --
# #SBATCH directives don't expand --export'd shell vars, only sbatch's own
# CLI flags do, so the real per-dataset/arm name is set there instead)
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_optuna_logs/%x_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_optuna_logs/%x_%j.err

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate

export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /lustre10/scratch/nourmaj/reve_optuna_logs
mkdir -p /lustre10/scratch/nourmaj/reve_optuna_results

# DATASET / ARM / FRACTION / N_TRIALS are passed in via
# sbatch --export=DATASET=...,ARM=...,FRACTION=...,N_TRIALS=... (see
# submit_reve_optuna_job.sh). Study state lives in a per-run sqlite db
# (load_if_exists=True), so a timed-out/re-submitted job just picks up
# where it left off instead of restarting trials from scratch.
python /lustre10/scratch/nourmaj/reve_optuna_search.py \
  --dataset "$DATASET" \
  --arm "$ARM" \
  --fraction "$FRACTION" \
  --n_trials "$N_TRIALS"
