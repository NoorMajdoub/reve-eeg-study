#!/bin/bash
#SBATCH --job-name=reve_chbmit_pretrained_50pct
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_chbmit_pretrained_50pct_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_chbmit_pretrained_50pct_%j.err

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre10/scratch/nourmaj/REVE/src
mkdir -p /lustre10/scratch/nourmaj/reve_outputs/chbmit_pretrained_50pct
ln -sfn /lustre10/scratch/nourmaj/hf_cache /lustre10/scratch/nourmaj/reve_outputs/chbmit_pretrained_50pct/.cache

# Best hyperparameters from reve_optuna_search.py (see
# chbmit_pretrained_50pct_best.json): val_balanced_acc=0.7118 over 14 trials.
python dt.py \
  task=chbmit \
  training_mode=ft \
  pretrained_path=hf:brain-bzh/reve-base \
  \
  task.data_loader.dataset.path=/lustre10/scratch/nourmaj/chb-mit-processed-80h_50pct \
  task.fine_tuning.optimizer.lr=6.0885333263570614e-05 \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=7 \
  task.fine_tuning.n_epochs=50 \
  trainer.clip_grad=null \
  seed=42 \
  hydra.run.dir=/lustre10/scratch/nourmaj/reve_outputs/chbmit_pretrained_50pct \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
