#!/bin/bash
#SBATCH --job-name=reve_caueeg_pretrained
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_caueeg_pretrained_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_caueeg_pretrained_%j.err

# Seed via: sbatch --export=ALL,SEED=2 reve_caueeg_pretrained.sbatch  (default 1)
SEED=${SEED:-1}

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=/lustre10/scratch/nourmaj/reve_outputs/caueeg_pretrained_seed${SEED}
cd /lustre10/scratch/nourmaj/REVE/src
mkdir -p "$OUT"
ln -sfn /lustre10/scratch/nourmaj/hf_cache "$OUT/.cache"

# Best hyperparameters from reve_optuna_search.py (caueeg_pretrained_100pct_best.json):
# val_balanced_acc=0.5378 over 7 trials. warmup_ratio=0.1029 -> round(0.1029*50)=5 epochs.
python dt.py \
  task=caueeg \
  training_mode=ft \
  pretrained_path=hf:brain-bzh/reve-base \
  task.data_loader.dataset.path=/lustre10/scratch/nourmaj/caueeg_reve_processed \
  task.fine_tuning.optimizer.lr=0.000169920420108681 \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=5 \
  task.fine_tuning.n_epochs=50 \
  trainer.clip_grad=0.5 \
  seed=${SEED} \
  hydra.run.dir="$OUT" \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
