#!/bin/bash
#SBATCH --job-name=reve_clinical_pretrained_25pct_seed2
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_clinical_pretrained_25pct_seed2_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_clinical_pretrained_25pct_seed2_%j.err

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre10/scratch/nourmaj/REVE/src
mkdir -p /lustre10/scratch/nourmaj/reve_outputs/clinical_pretrained_25pct_seed2
ln -sfn /lustre10/scratch/nourmaj/hf_cache /lustre10/scratch/nourmaj/reve_outputs/clinical_pretrained_25pct_seed2/.cache

# Best hyperparameters from reve_optuna_search.py (see
# clinical_pretrained_25pct_best.json): val_balanced_acc=0.7322 over 13 trials.
python dt.py \
  task=clinical \
  training_mode=ft \
  pretrained_path=hf:brain-bzh/reve-base \
  \
  task.data_loader.dataset.path=/lustre10/scratch/nourmaj/ds004504_processed_25pct \
  task.fine_tuning.optimizer.lr=5.5819039355173704e-05 \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=0 \
  task.fine_tuning.n_epochs=50 \
  trainer.clip_grad=0.5 \
  seed=2 \
  hydra.run.dir=/lustre10/scratch/nourmaj/reve_outputs/clinical_pretrained_25pct_seed2 \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
