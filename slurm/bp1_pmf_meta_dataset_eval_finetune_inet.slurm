#!/bin/bash
#SBATCH --job-name=pmf_md_eval_finetune_inet
#SBATCH --partition=gpu
#SBATCH --account=coms037985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_3090:4
#SBATCH --nodelist=bp1-gpu030
#SBATCH --time=48:00:00
#SBATCH --output=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.out
#SBATCH --error=/user/home/im23926/bsc-dissertation-lora-cdfsl/logs/slurm/slurm_%x_%j.err

echo "Host: $(hostname)"
echo "Start: $(date)"
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
CKPT="$REPO_ROOT/checkpoints/pmf_metadataset_dino/md_imagenet_128x128_dinosmall_fp16_lr2e-4/best.pth"
OUTDIR="$REPO_ROOT/logs/pmf/meta_dataset_inet/eval/${SLURM_JOB_ID}"

echo "PMF_DIR=$PMF_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "CKPT=$CKPT"
echo "OUTDIR=$OUTDIR"

mkdir -p "$OUTDIR" && echo "mkdir ok" || echo "mkdir FAILED"
test -d "$PMF_DIR" && echo "PMF_DIR ok" || echo "PMF_DIR FAILED"
test -d "$DATA_PATH" && echo "DATA_PATH ok" || echo "DATA_PATH FAILED"
test -f "$CKPT" && echo "CKPT ok" || echo "CKPT FAILED"
ulimit -n 100000 && echo "ulimit ok" || echo "ulimit FAILED"
cd "$PMF_DIR" && echo "cd ok" || echo "cd FAILED"

echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo "Starting torchrun..."
torchrun --standalone --nproc_per_node=4 test_meta_dataset.py \
  --data-path "$DATA_PATH" \
  --dataset meta_dataset \
  --arch dino_small_patch16 \
  --deploy finetune \
  --output "$OUTDIR" \
  --resume "$CKPT" \
  --dist-eval \
  --ada_steps 100 \
  --ada_lr 0.0001 \
  --aug_prob 0.9 \
  --aug_types color translation

echo "End: $(date)"