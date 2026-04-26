#!/bin/bash
#SBATCH --job-name=debug_isambard
#SBATCH --partition=workq
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/debug_isambard_%j.out

set +u
source ~/miniforge3/bin/activate
export xml_catalog_files_libxml2=""
conda activate pmf
set -u

module load cuda/12.6

cd ~/bsc-dissertation-lora-cdfsl/src/pmf_cvpr22
ulimit -n 100000

python test_meta_dataset.py \
    --dataset meta_dataset \
    --data-path /lus/lfs1aip2/scratch/u6hz/frixos.u6hz/data/meta_dataset_h5 \
    --arch dino_small_patch16 \
    --deploy vanilla \
    --resume /lus/lfs1aip2/scratch/u6hz/frixos.u6hz/checkpoints/pmf_metadataset_dino/md_full_128x128_dinosmall_fp16_lr5e-5/best.pth \
    --test_sources ilsvrc_2012 \
    --nEpisode 20 \
    --fp16 \
    --device cuda:0 \
    --output /lus/lfs1aip2/scratch/u6hz/frixos.u6hz/bsc-dissertation-lora-cdfsl/logs/debug
