#!/bin/bash
#SBATCH --job-name=dataset_audit
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/nourmaj/dataset_audit_%j.out
#SBATCH --error=/scratch/nourmaj/dataset_audit_%j.err
# Read-only audit of every processed dataset folder (CPU only). Rorqual.
#   sbatch audit_datasets.sbatch
source /scratch/nourmaj/envs/eeg-gpu/bin/activate
python /scratch/nourmaj/audit_datasets.py --workers 8
