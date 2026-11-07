#!/bin/bash
#SBATCH --job-name=prep_things_eeg2_reve
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre10/scratch/nourmaj/prep_things_eeg2_reve_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/prep_things_eeg2_reve_%j.err

module load StdEnv/2023 python/3.11
source /lustre10/scratch/nourmaj/envs/prep/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

python /lustre10/scratch/nourmaj/preprocess_things_eeg2_reve.py \
  --root /lustre10/scratch/nourmaj/things_eeg2 \
  --out_dir /lustre10/scratch/nourmaj/things_eeg2_reve_processed
