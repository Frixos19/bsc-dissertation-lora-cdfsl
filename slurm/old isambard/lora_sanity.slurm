#!/bin/bash
#SBATCH --job-name=lora_sanity
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

set +u
source ~/miniforge3/bin/activate
conda activate pmf
export xml_catalog_files_libxml2=""
set -u

export PATH=$(echo "$PATH" | sed 's|/opt/cray/pe/python/[^/]*/bin:||')
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r

PMF_DIR="/scratch/u5hv/frixos.u5hv/pmf_cvpr22"
cd "$PMF_DIR"

python - << 'EOF'
import argparse
from models import get_backbone
from models.deploy import ProtoNet_LoRA_Finetune
from models.lora import LoRALinear

args = argparse.Namespace(arch='dino_small_patch16', no_pretrain=False)

backbone = get_backbone(args)
model = ProtoNet_LoRA_Finetune(backbone, r=4, lora_alpha=1)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
lora_mods = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
non_lora  = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]

print(f"Trainable:  {trainable:,}")
print(f"Frozen:     {frozen:,}")
print(f"LoRA modules injected: {len(lora_mods)}")
print(f"Non-LoRA trainable (should be empty): {non_lora}")

assert len(lora_mods) == 12, f"Expected 12, got {len(lora_mods)}"
assert len(non_lora) == 0,   f"Non-LoRA params are trainable: {non_lora}"
assert trainable == 73728,   f"Expected 73728, got {trainable}"
print("ALL CHECKS PASSED")
EOF