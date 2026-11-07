#!/bin/bash
#SBATCH --job-name=dataset_audit2
#SBATCH --account=def-amiilab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/nourmaj/dataset_audit2_%j.out
#SBATCH --error=/scratch/nourmaj/dataset_audit2_%j.err
# Read-only dataset audit, pass 2 (Rorqual). Uses the REVE env because it has lmdb.
#   sbatch audit_pass2.sbatch
source /scratch/nourmaj/envs/reve-gpu/bin/activate 2>/dev/null || source /scratch/nourmaj/envs/eeg-gpu/bin/activate
python /scratch/nourmaj/audit_pass2.py
