#!/bin/bash
#SBATCH --job-name=download_cifar_mini
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.err

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"

SCRIPTS_DIR="$HOME/bsc-dissertation-lora-cdfsl/src/pmf_cvpr22/scripts"

cd "$SCRIPTS_DIR"
sh download_cifarfs.sh
sh download_miniimagenet.sh

echo "Done: $(date)"
ls ../data/cifar-fs/
ls ../data/Mini-ImageNet/
