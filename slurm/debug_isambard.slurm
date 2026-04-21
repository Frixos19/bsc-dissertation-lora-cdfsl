#!/bin/bash
#SBATCH --job-name=debug_isambard
#SBATCH --partition=workq
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/debug_isambard_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate pmf

module load cuda/11.8

cd ~/bsc-dissertation-lora-cdfsl/src/pmf_cvpr22

ulimit -n 100000

python test_meta_dataset.py \
    --dataset meta_dataset \
    --data-path /lus/lfs1aip2/scratch/b6aq/frixos.b6aq/data/meta_dataset_h5 \
    --pretrained-checkpoint-path /lus/lfs1aip2/scratch/b6aq/frixos.b6aq/checkpoints/pmf_metadataset_dino \
    --checkpoint md_full_128x128_dinosmall_fp16_lr5e-5/best.pth \
    --arch dino_small_patch16 \
    --test_sources ilsvrc_2012 \
    --nEpisode 20 \
    --fp16 \
    --device cuda:0
