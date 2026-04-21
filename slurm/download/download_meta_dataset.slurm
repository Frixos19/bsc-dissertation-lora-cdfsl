#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

echo "Start: $(date)"

git lfs install

cd /lus/lfs1aip2/scratch/b6aq/frixos.b6aq/data
git clone https://huggingface.co/datasets/hushell/meta_dataset_h5

echo "End: $(date)"
