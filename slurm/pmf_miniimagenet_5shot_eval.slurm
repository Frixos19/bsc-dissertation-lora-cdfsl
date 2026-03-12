#!/bin/bash
#SBATCH --job-name=pmf_mini_5s_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
source ~/miniforge3/bin/activate
conda activate pmf
export xml_catalog_files_libxml2=""
set -u

export PATH=$(echo "$PATH" | sed 's|/opt/cray/pe/python/[^/]*/bin:||')
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

echo "Python: $(which python)"
python -V

REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

./scripts/bootstrap_storage.sh

PMF_DIR="/scratch/u5hv/frixos.u5hv/pmf_cvpr22"
if [ ! -d "$PMF_DIR" ]; then
  echo "ERROR: PMF_DIR not found: $PMF_DIR"
  exit 1
fi

TRAIN_JOBID=2219783
CKPT="$REPO_ROOT/logs/pmf/mini_imagenet/5shot/train/${TRAIN_JOBID}/best.pth"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

OUTDIR="$REPO_ROOT/logs/pmf/mini_imagenet/5shot/eval/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "PMF_DIR=$PMF_DIR"
echo "OUTDIR=$OUTDIR"
echo "CKPT=$CKPT"

cd "$PMF_DIR"

# --- MiniImageNet 5-way 5-shot EVAL ---
python main.py \
  --output "$OUTDIR" \
  --dataset mini_imagenet \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --deploy finetune_autolr \
  --device cuda:0 \
  --nSupport 5 \
  --fp16 \
  --resume "$CKPT" \
  --nEpisode 600 \
  --eval

echo "End: $(date)"