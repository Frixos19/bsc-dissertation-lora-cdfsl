#!/bin/bash
#SBATCH --job-name=extract_md
#SBATCH --partition=workq
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/bsc-dissertation-lora-cdfsl/logs/extract_meta_dataset_%j.out

DATA=/lus/lfs1aip2/scratch/b6aq/frixos.b6aq/data/meta_dataset_h5

cd $DATA

for f in *.tar.gz; do
    echo "Extracting $f..."
    tar -xzf $f --checkpoint=1000 --checkpoint-action=echo="%T"
    echo "Done: $f"
done

echo "All done. Final contents:"
ls -lh $DATA
