#!/bin/bash
#SBATCH --job-name=pmf_md_ckpt
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u5hv/frixos.u5hv/md_ckpt_%j.out
#SBATCH --error=/scratch/u5hv/frixos.u5hv/md_ckpt_%j.err

set -euo pipefail

# --- Conda ---
set +u
source ~/miniforge3/bin/activate
export xml_catalog_files_libxml2=""
conda activate pmf
set -u

DEST="/scratch/u5hv/frixos.u5hv/pmf_metadataset_dino"

python - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hushell/pmf_metadataset_dino",
    local_dir="/scratch/u5hv/frixos.u5hv/pmf_metadataset_dino",
    local_dir_use_symlinks=False,
)
print("Checkpoint download complete.")
EOF

