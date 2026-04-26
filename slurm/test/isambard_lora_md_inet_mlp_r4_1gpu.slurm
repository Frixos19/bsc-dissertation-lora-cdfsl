#!/bin/bash
#SBATCH --job-name=isambard_lora_md_inet_mlp_r4_1gpu
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/slurm_%x_%j.err

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

echo "Python: $(which python)"
python -V
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"

REPO_ROOT="$HOME/bsc-dissertation-lora-cdfsl"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
DATA_PATH="/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/data/meta_dataset_h5"
CKPT="/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/checkpoints/pmf_metadataset_dino/md_imagenet_128x128_dinosmall_fp16_lr2e-4/best.pth"
OUTDIR="/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/lora/meta_dataset_inet/eval_mlp_r4/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
test -d "$PMF_DIR" && echo "PMF_DIR ok" || { echo "PMF_DIR FAILED"; exit 1; }
test -d "$DATA_PATH" && echo "DATA_PATH ok" || { echo "DATA_PATH FAILED"; exit 1; }
test -f "$CKPT" && echo "CKPT ok" || { echo "CKPT FAILED"; exit 1; }
ulimit -n 100000
cd "$PMF_DIR"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Starting..."
python test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune_lora \
  --output "$OUTDIR" \
  --resume "$CKPT" \
  --ada_steps 50 \
  --ada_lr 0.0001 \
  --aug_prob 0.9 \
  --aug_types color translation \
  --lora_r 4 \
  --lora_alpha 1 \
  --lora_target mlp

echo "End: $(date)"
