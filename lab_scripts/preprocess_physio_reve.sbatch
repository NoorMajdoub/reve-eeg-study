#!/bin/bash
#SBATCH --job-name=prep_physio_reve
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --tmp=40G
#SBATCH --output=/lustre10/scratch/nourmaj/prep_physio_reve_%j.out
#SBATCH --error=/lustre10/scratch/nourmaj/prep_physio_reve_%j.err

# REVE authors' PhysioNet-MI preprocessing, unchanged -- but the LMDB is written
# to the node's local SSD ($SLURM_TMPDIR) and copied to scratch at the end.
# (The script commits one transaction per window, which is very slow on Lustre.)

set -euo pipefail
module load StdEnv/2023 python/3.11
source /lustre10/scratch/nourmaj/envs/prep/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

cd /lustre10/scratch/nourmaj/REVE
python preprocessing/preprocessing_physio.py \
  --root /lustre10/scratch/nourmaj/raw_eegmmidb \
  --processed "$SLURM_TMPDIR/physio_reve_processed" \
  --type reve

rm -rf /lustre10/scratch/nourmaj/physio_reve_processed
cp -r "$SLURM_TMPDIR/physio_reve_processed" /lustre10/scratch/nourmaj/physio_reve_processed
echo "===== DONE: $(du -sh /lustre10/scratch/nourmaj/physio_reve_processed)"
