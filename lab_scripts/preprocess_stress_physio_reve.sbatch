#!/bin/bash
#SBATCH --job-name=prep_stress_physio_reve
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre10/scratch/nourmaj/prep_stress_physio_reve_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/prep_stress_physio_reve_%j.err

# Runs the REVE authors' own preprocessing scripts (unchanged) for
# Mental Arithmetic (eegmat) and PhysioNet-MI (eegmmidb).

module load StdEnv/2023 python/3.11
source /lustre10/scratch/nourmaj/envs/prep/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd /lustre10/scratch/nourmaj/REVE

echo "===== STRESS"
python preprocessing/preprocessing_stress.py \
  --root /lustre10/scratch/nourmaj/raw_eegmat \
  --processed /lustre10/scratch/nourmaj/stress_reve_processed \
  --type reve

echo "===== PHYSIO"
python preprocessing/preprocessing_physio.py \
  --root /lustre10/scratch/nourmaj/raw_eegmmidb \
  --processed /lustre10/scratch/nourmaj/physio_reve_processed \
  --type reve

echo "===== DONE"
