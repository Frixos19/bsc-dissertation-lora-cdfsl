#!/bin/bash
#SBATCH --job-name=eval_cifarfs_5s_lora
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.err

set -euo pipefail
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

set +u
source ~/miniforge3/bin/activate
export xml_catalog_files_libxml2=""
conda activate pmf
set -u

module load cuda/12.6

REPO_ROOT="$HOME/bsc-dissertation-lora-cdfsl"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
TRAIN_JOBID="3764574"
CKPT="/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/pmf/cifarfs/5shot/train/${TRAIN_JOBID}/best.pth"
OUTDIR="/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/pmf/cifarfs/5shot/eval_lora_r4/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
test -d "$PMF_DIR" && echo "PMF_DIR ok" || { echo "PMF_DIR FAILED"; exit 1; }
test -f "$CKPT" && echo "CKPT ok" || { echo "CKPT not found: $CKPT"; exit 1; }

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
  --ada_steps 100 \
  --ada_lr 1e-5 \
  --aug_prob 0.9 \
  --aug_types color translation \
  --lora_r 4 \
  --lora_alpha 4

echo "End: $(date)"
