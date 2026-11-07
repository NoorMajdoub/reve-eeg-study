#!/bin/bash
#SBATCH --job-name=reve2cbramod
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/reve2cbramod_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/reve2cbramod_%j.err

module load StdEnv/2023 python/3.11
source /lustre10/scratch/nourmaj/envs/prep/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python /lustre10/scratch/nourmaj/convert_reve_to_cbramod.py --task sleepedf \
  --src /lustre10/scratch/nourmaj/sleepedf_reve_processed \
  --dst /lustre10/scratch/nourmaj/sleepedf_cbramod_from_reve --workers 16

python /lustre10/scratch/nourmaj/convert_reve_to_cbramod.py --task caueeg \
  --src /lustre10/scratch/nourmaj/caueeg_reve_processed \
  --dst /lustre10/scratch/nourmaj/caueeg_cbramod_from_reve --workers 16
