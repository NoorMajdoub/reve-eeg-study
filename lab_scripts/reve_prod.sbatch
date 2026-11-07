#!/bin/bash
#SBATCH --job-name=reve_prod
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_prod_%x_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_prod_%x_%j.err

# Generic REVE production run: reads the best hyperparameters straight from the
# Optuna result file, so nothing is copied by hand.
#   sbatch --job-name=isruc_pre_25_s1 \
#     --export=ALL,TASK=isruc,ARM=pretrained,FRACTION=25,SEED=1,DATA=/lustre10/scratch/nourmaj/isruc_processed_80h_25pct \
#     reve_prod.sbatch
# Output folder: reve_outputs/<task>_<arm>_<fraction>pct_seed<seed>

set -euo pipefail
: "${TASK:?}" "${ARM:?}" "${FRACTION:?}" "${SEED:?}" "${DATA:?}"
N_EPOCHS=${N_EPOCHS:-50}
BEST=/lustre10/scratch/nourmaj/reve_optuna_results/${TASK}_${ARM}_${FRACTION}pct_best.json

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

read LR CLIP WARMUP < <(python -c "
import json; p = json.load(open('$BEST'))['best_params']
clip = 'null' if p['grad_clip'] is None else p['grad_clip']
print(p['lr'], clip, round(p['warmup_ratio'] * $N_EPOCHS))
")
if [ "$ARM" = "pretrained" ]; then PP=hf:brain-bzh/reve-base; else PP=""; fi
echo "best params from $BEST: lr=$LR clip=$CLIP warmup_epochs=$WARMUP"

OUT=/lustre10/scratch/nourmaj/reve_outputs/${TASK}_${ARM}_${FRACTION}pct_seed${SEED}
mkdir -p "$OUT"
ln -sfn /lustre10/scratch/nourmaj/hf_cache "$OUT/.cache"
cd /lustre10/scratch/nourmaj/REVE/src

python dt.py \
  task=${TASK} \
  training_mode=ft \
  pretrained_path=${PP} \
  task.data_loader.dataset.path=${DATA} \
  task.fine_tuning.optimizer.lr=${LR} \
  task.fine_tuning.optimizer.weight_decay=1e-05 \
  task.fine_tuning.warmup_epochs=${WARMUP} \
  task.fine_tuning.n_epochs=${N_EPOCHS} \
  trainer.clip_grad=${CLIP} \
  seed=${SEED} \
  hydra.run.dir="$OUT" \
  cache_dir=/lustre10/scratch/nourmaj/hf_cache
