#!/bin/bash
#SBATCH --job-name=md_inet_1shot_lora_r1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
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

echo "Python: $(which python)"
python -V
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"

REPO_ROOT="$HOME/bsc-dissertation-lora-cdfsl"
PMF_DIR="$REPO_ROOT/src/pmf_cvpr22"
DATA_PATH="/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/data/meta_dataset_h5"
CKPT="/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/checkpoints/pmf_metadataset_dino/md_imagenet_128x128_dinosmall_fp16_lr2e-4/best.pth"
OUTDIR="/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/shot/meta_dataset_inet/1shot/eval_r2/${SLURM_JOB_ID}"

mkdir -p "$OUTDIR"
test -d "$PMF_DIR" && echo "PMF_DIR ok" || { echo "PMF_DIR FAILED"; exit 1; }
test -d "$DATA_PATH" && echo "DATA_PATH ok" || { echo "DATA_PATH FAILED"; exit 1; }
test -f "$CKPT" && echo "CKPT ok" || { echo "CKPT FAILED"; exit 1; }
ulimit -n 100000
cd "$PMF_DIR"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Starting torchrun..."

torchrun --standalone --nproc_per_node=4 test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune_lora \
  --output_dir "$OUTDIR" \
  --resume "$CKPT" \
  --ada_steps 100 \
  --ada_lr 0.0001 \
  --aug_prob 0.9 \
  --aug_types color translation \
  --dist-eval \
  --lora_r 1 \
  --lora_alpha 1 \
  --lora_target qkv \
  --num_support 1 \
  --nEpisode 600

echo "End: $(date)"
