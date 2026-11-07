#!/bin/bash
#SBATCH --job-name=reve_japaneeg_random
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_japaneeg_random_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_japaneeg_random_%j.err

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre10/scratch/nourmaj/REVE/src
mkdir -p /lustre10/scratch/nourmaj/reve_outputs/japaneeg_random
ln -sfn /lustre10/scratch/nourmaj/hf_cache /lustre10/scratch/nourmaj/reve_outputs/japaneeg_random/.cache

# Best hyperparameters from reve_optuna_search.py (see
# japaneeg_random_100pct_best.json): val_balanced_acc=0.5453 over 20 trials.
python dt.py \
  task=japaneeg \
  training_mode=ft \
  pretrained_path="" \
  \
  task.data_loader.dataset.path=/lustre10/scratch/nourmaj/japaneeg_processed_80h \
  task.fine_tuning.optimizer.lr=0.00010751651135740108 \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=7 \
  task.fine_tuning.n_epochs=50 \
  trainer.clip_grad=null \
  seed=42 \
  hydra.run.dir=/lustre10/scratch/nourmaj/reve_outputs/japaneeg_random \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
