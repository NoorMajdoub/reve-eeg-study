#!/bin/bash
#SBATCH --job-name=reve_diag_bci2020
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_smoke_logs/%x_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_smoke_logs/%x_%j.err

# Diagnostic, NOT a benchmark run: is our BCI2020 pipeline capable of reaching
# the published range (~50%+) when trained the way REVE's authors train it
# (configs/task/speech.yaml: batch 64, patience 15, lr 1e-4, long training)?
#   - reaches ~40-55%  -> data pipeline is fine; the gap is our patience-2 protocol
#   - stays ~20-27%    -> something in the data pipeline is wrong
# Usage: sbatch --export=ALL,ARM=pretrained reve_diag_bci2020.sbatch   (or ARM=random)

ARM=${ARM:-pretrained}
if [ "$ARM" = "pretrained" ]; then PP=hf:brain-bzh/reve-base; else PP=""; fi
DATA=${DATA:-/lustre10/scratch/nourmaj/bci2020_reve_processed}
TAG=${TAG:-default}
LR=${LR:-1e-4}
SEED=${SEED:-1}
MIXUP=${MIXUP:-True}

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT=/lustre10/scratch/nourmaj/reve_diag_outputs/bci2020_${ARM}_${TAG}
mkdir -p "$OUT"
ln -sfn /lustre10/scratch/nourmaj/hf_cache "$OUT/.cache"
cd /lustre10/scratch/nourmaj/REVE/src

python dt.py \
  task=bci2020 \
  training_mode=ft \
  pretrained_path=${PP} \
  task.data_loader.dataset.path=${DATA} \
  task.data_loader.batch_size=64 \
  task.fine_tuning.n_epochs=150 \
  task.fine_tuning.patience=15 \
  task.fine_tuning.warmup_epochs=5 \
  task.fine_tuning.optimizer.lr=${LR} \
  task.fine_tuning.mixup=${MIXUP} \
  seed=${SEED} \
  hydra.run.dir="$OUT" \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
