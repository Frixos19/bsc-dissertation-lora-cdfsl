#!/bin/bash
#SBATCH --job-name=pmf_md_eval
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

echo "== Job info =="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +u
source ~/work/miniforge3/bin/activate
conda activate pmf
set -u

echo "Python: $(which python)"
python -V
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
DATA_PATH="$REPO_ROOT/data/meta_dataset_h5"
CKPT="$REPO_ROOT/checkpoints/pmf_metadataset_dino/md_full_128x128_dinosmall_fp16_lr5e-5/best.pth"

OUTDIR="$REPO_ROOT/logs/pmf/meta_dataset/eval/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "PMF_DIR=$PMF_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "CKPT=$CKPT"
echo "OUTDIR=$OUTDIR"

test -d "$PMF_DIR"
test -d "$DATA_PATH"
test -f "$CKPT"

ulimit -n 100000

cd "$PMF_DIR"

torchrun --standalone --nproc_per_node=4 test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune_autolr \
  --output "$OUTDIR" \
  --resume "$CKPT" \
  --dist-eval \
  --ada_steps 100

echo "End: $(date)"
