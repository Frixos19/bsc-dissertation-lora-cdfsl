#!/bin/bash
#SBATCH --job-name=md_extract
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

ROOT="/scratch/u5hv/frixos.u5hv/data/meta_dataset_h5"
cd "$ROOT"

echo "Extracting tarballs in $ROOT"
for f in *.tar.gz; do
  echo "== Extracting $f =="
  tar -xzf "$f"
done

echo "Done. Checking a few domains:"
for d in traffic_sign mscoco omniglot aircraft; do
  echo "--- $d ---"
  ls -l "$d/dataset_spec.json"
done

echo "End: $(date)"

