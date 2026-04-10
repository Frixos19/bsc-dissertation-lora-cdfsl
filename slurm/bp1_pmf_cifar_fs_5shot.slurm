#!/bin/bash
#SBATCH --job-name=pmf_cifarfs_5s
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

echo "== Job info =="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +u
source ~/work/miniforge3/bin/activate
conda activate pmf
set -u

echo "Python: $(which python)"
python -V

REPO_ROOT="$SLURM_SUBMIT_DIR"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"

OUTDIR="$REPO_ROOT/logs/pmf/cifar_fs/5shot/train/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "PMF_DIR=$PMF_DIR"
echo "OUTDIR=$OUTDIR"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$PMF_DIR"

python main.py \
  --output "$OUTDIR" \
  --dataset cifar_fs \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --device cuda:0 \
  --nSupport 5 \
  --fp16

echo "End: $(date)"
