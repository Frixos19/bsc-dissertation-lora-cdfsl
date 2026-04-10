#!/bin/bash
#SBATCH --job-name=lora_cifarfs_5s
#SBATCH --partition=gpu_short
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.out
#SBATCH --error=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"

set +u
source ~/work/miniforge3/bin/activate
conda activate pmf
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
CKPT="$REPO_ROOT/logs/pmf/cifar_fs/5shot/train/16328650/best.pth"
OUTDIR="$REPO_ROOT/logs/lora/cifar_fs/5shot/eval/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
cd "$PMF_DIR"

python main.py \
  --output "$OUTDIR" \
  --dataset cifar_fs \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --deploy finetune_lora \
  --device cuda:0 \
  --nSupport 5 \
  --fp16 \
  --resume "$CKPT" \
  --nEpisode 600 \
  --eval \
  --lora_r 4 \
  --lora_alpha 1 \
  --ada_steps 100 \
  --ada_lr 0.001 \
  --aug_prob 0.9 \
  --aug_types color translation

echo "End: $(date)"