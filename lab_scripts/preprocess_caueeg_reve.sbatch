#!/bin/bash
#SBATCH --job-name=prep_caueeg_reve
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre10/scratch/nourmaj/prep_caueeg_reve_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/prep_caueeg_reve_%j.err

module load StdEnv/2023 python/3.11
source /lustre10/scratch/nourmaj/envs/prep/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python /lustre10/scratch/nourmaj/preprocess_caueeg_reve.py \
  --caueeg_dir /lustre10/scratch/nourmaj/caueeg/caueeg-dataset \
  --out_dir /lustre10/scratch/nourmaj/caueeg_reve_processed \
  --workers 16
