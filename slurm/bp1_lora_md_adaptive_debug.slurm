#!/bin/bash
#SBATCH --job-name=lora_md_adaptive_debug
#SBATCH --partition=gpu_short
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=00:15:00
#SBATCH --output=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.out
#SBATCH --error=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.err

set +u
source ~/work/miniforge3/bin/activate
conda activate pmf
set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

REPO_ROOT="$SLURM_SUBMIT_DIR"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
DATA_PATH="$REPO_ROOT/data/meta_dataset_h5"
CKPT="$REPO_ROOT/checkpoints/pmf_metadataset_dino/md_full_128x128_dinosmall_fp16_lr5e-5/best.pth"
OUTDIR="$REPO_ROOT/logs/lora/meta_dataset/eval_adaptive_debug/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
cd "$PMF_DIR"

python test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune_lora_adaptive \
  --output "$OUTDIR" \
  --resume "$CKPT" \
  --ada_steps 100 \
  --ada_lr 0.0001 \
  --aug_prob 0.9 \
  --aug_types color translation \
  --lora_r 4 \
  --lora_alpha 1 \
  --nEpisode 10

echo "End: $(date)"
