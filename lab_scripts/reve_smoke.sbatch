#!/bin/bash
#SBATCH --job-name=reve_smoke
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_smoke_logs/%x_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_smoke_logs/%x_%j.err

# One-epoch end-to-end check of a new REVE task (pretrained arm).
# Usage: sbatch --job-name=reve_smoke_sleepedf --export=ALL,TASK=sleepedf reve_smoke.sbatch

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=/lustre10/scratch/nourmaj/reve_smoke_outputs/${TASK}
mkdir -p "$OUT"
ln -sfn /lustre10/scratch/nourmaj/hf_cache "$OUT/.cache"
cd /lustre10/scratch/nourmaj/REVE/src

start=$(date +%s)
python dt.py \
  task=${TASK} \
  training_mode=ft \
  pretrained_path=hf:brain-bzh/reve-base \
  task.fine_tuning.n_epochs=1 \
  task.fine_tuning.warmup_epochs=0 \
  seed=1 \
  hydra.run.dir="$OUT" \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
echo "SMOKE_WALLTIME_SECONDS=$(( $(date +%s) - start ))"
