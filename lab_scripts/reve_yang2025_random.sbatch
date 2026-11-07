#!/bin/bash
#SBATCH --job-name=reve_yang2025_random
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_yang2025_random_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_yang2025_random_%j.err

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /lustre10/scratch/nourmaj/REVE/src
mkdir -p /lustre10/scratch/nourmaj/reve_outputs/yang2025_random
ln -sfn /lustre10/scratch/nourmaj/hf_cache /lustre10/scratch/nourmaj/reve_outputs/yang2025_random/.cache

# Best hyperparameters from reve_optuna_search.py (see
# yang2025_random_100pct_best.json): val_balanced_acc=0.8223 over 20 trials.
python dt.py \
  task=yang2025 \
  training_mode=ft \
  pretrained_path="" \
  \
  task.data_loader.dataset.path=/lustre10/scratch/nourmaj/yang2025_processed \
  task.fine_tuning.optimizer.lr=0.00017190806866086445 \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=3 \
  task.fine_tuning.n_epochs=50 \
  trainer.clip_grad=null \
  seed=2 \
  hydra.run.dir=/lustre10/scratch/nourmaj/reve_outputs/yang2025_random \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
