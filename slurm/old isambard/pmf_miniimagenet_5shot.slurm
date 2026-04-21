#!/bin/bash
#SBATCH --job-name=pmf_mini_5s
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
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# --- Init conda (miniforge) ---
set +u # conda activate/deactivate hooks can reference unset vars

source ~/miniforge3/bin/activate
conda activate pmf

# Avoid libxml2 hook nounset crash
export xml_catalog_files_libxml2=""

set -u  # re-enable nounset

# --- FIX: Cray python can appear before conda python on PATH ---
export PATH=$(echo "$PATH" | sed 's|/opt/cray/pe/python/[^/]*/bin:||')
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

echo "Python: $(which python)"
python -V

# --- Your dissertation repo root ---
REPO_ROOT="$SLURM_SUBMIT_DIR"
cd "$REPO_ROOT"

# Ensure scratch symlinks exist (safe to rerun)
./scripts/bootstrap_storage.sh

# --- PMF repo path (EDIT THIS) ---
PMF_DIR="/scratch/u5hv/frixos.u5hv/pmf_cvpr22"

if [ ! -d "$PMF_DIR" ]; then
  echo "ERROR: PMF_DIR not found: $PMF_DIR"
  exit 1
fi

# --- Output directory in scratch-backed logs ---
OUTDIR="$REPO_ROOT/logs/pmf/mini_imagenet/5shot/${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "PMF_DIR=$PMF_DIR"
echo "OUTDIR=$OUTDIR"

cd "$PMF_DIR"

# --- Run PMF: MiniImageNet 5-shot ---
python main.py \
  --output "$OUTDIR" \
  --dataset mini_imagenet \
  --epoch 100 \
  --lr 5e-5 \
  --arch dino_small_patch16 \
  --device cuda:0 \
  --nSupport 5 \
  --fp16

echo "End: $(date)"
