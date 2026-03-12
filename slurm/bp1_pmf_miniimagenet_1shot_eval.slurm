#!/bin/bash
#SBATCH --job-name=pmf_mini_1s_eval
#SBATCH --partition=gpu_short
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=01:00:00
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

TRAIN_JOBID=2224852
CKPT="$REPO_ROOT/logs/pmf/mini_imagenet/1shot/train/${TRAIN_JOBID}/best.pth"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

OUTDIR="$REPO_ROOT/logs/pmf/mini_imagenet/1shot/eval/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

cd "$PMF_DIR"

python main.py \
  --output "$OUTDIR" \
  --dataset mini_imagenet \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --deploy finetune_autolr \
  --device cuda:0 \
  --nSupport 1 \
  --fp16 \
  --resume "$CKPT" \
  --nEpisode 600 \
  --eval

echo "End: $(date)"
