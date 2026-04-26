#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.err

echo "Start: $(date)"

git lfs install

cd /lus/lfs1aip2/scratch/u6hz/frixos.u6hz/data
git clone https://huggingface.co/datasets/hushell/meta_dataset_h5

echo "End: $(date)"
