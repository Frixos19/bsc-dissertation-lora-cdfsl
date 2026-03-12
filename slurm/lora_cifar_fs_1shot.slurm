#!/bin/bash
#SBATCH --job-name=lora_cifarfs_1s
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

echo "== Job info =="
echo "Host: $(hostname)"
echo "Start: $(date)"
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
cd "$PMF_DIR"

OUTDIR="$REPO_ROOT/logs/lora/cifar_fs/1shot/r4/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
echo "OUTDIR=$OUTDIR"

python main.py \
  --output "$OUTDIR" \
  --dataset cifar_fs \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --device cuda:0 \
  --nSupport 1 \
  --fp16 \
  --deploy finetune_lora \
  --lora_r 4 \
  --lora_alpha 1

echo "End: $(date)"