#!/bin/bash
#SBATCH --job-name=reve_xfer
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve_xfer_logs/%x_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve_xfer_logs/%x_%j.err

# Runs all seeds for one (backbone, target) pair inside one job.
#   sbatch --job-name=xfer_clinical_isruc \
#     --export=ALL,TARGET=isruc,INIT=source,SOURCE_NAME=clinical,SOURCE_CKPT=/path/model_best.pth,SEEDS="1 2 3" \
#     reve_xfer.sbatch
#   INIT=pretrained or INIT=random need no SOURCE_*.

set -euo pipefail
: "${TARGET:?}" "${INIT:?}"
SEEDS=${SEEDS:-"1 2 3"}
SOURCE_NAME=${SOURCE_NAME:-none}
SOURCE_CKPT=${SOURCE_CKPT:-}

source /lustre10/scratch/nourmaj/envs/reve-gpu/bin/activate
export MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp XDG_CACHE_HOME=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/lustre10/scratch/nourmaj/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for SEED in $SEEDS; do
  echo "===== ${INIT} ${SOURCE_NAME} -> ${TARGET} seed ${SEED}"
  python /lustre10/scratch/nourmaj/reve_xfer_lp.py \
    --target "$TARGET" --init "$INIT" --source_name "$SOURCE_NAME" \
    ${SOURCE_CKPT:+--source_ckpt "$SOURCE_CKPT"} --seed "$SEED"
done
