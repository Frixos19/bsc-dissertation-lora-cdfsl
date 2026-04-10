#!/bin/bash
#SBATCH --job-name=lora_md_inet_mlp_r4
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_3090:4
#SBATCH --time=96:00:00
#SBATCH --output=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.out
#SBATCH --error=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

set +u
source ~/work/miniforge3/bin/activate
conda activate pmf
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
DATA_PATH="$REPO_ROOT/data/meta_dataset_h5"
CKPT="$REPO_ROOT/checkpoints/pmf_metadataset_dino/md_imagenet_128x128_dinosmall_fp16_lr2e-4/best.pth"
OUTDIR="$REPO_ROOT/logs/lora/meta_dataset_inet/eval/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
test -d "$PMF_DIR" && echo "PMF_DIR ok" || echo "PMF_DIR FAILED"
test -d "$DATA_PATH" && echo "DATA_PATH ok" || echo "DATA_PATH FAILED"
test -f "$CKPT" && echo "CKPT ok" || echo "CKPT FAILED"
ulimit -n 100000
cd "$PMF_DIR"

echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Starting torchrun..."

torchrun --standalone --nproc_per_node=4 test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune_lora \
  --output "$OUTDIR" \
  --resume "$CKPT" \
  --ada_steps 100 \
  --ada_lr 0.0001 \
  --aug_prob 0.9 \
  --aug_types color translation \
  --lora_r 4 \
  --lora_alpha 1 \
  --lora_target mlp \
  --dist-eval

echo "End: $(date)"
