# bsc-dissertation-lora-cdfsl
# BSc Dissertation: Parameter-Efficient Few-Shot Learning with LoRA

**Author:** Frixos  
**Institution:** University of Bristol   
**Deadline:** May 5th, 2026

## Overview

Investigating whether LoRA (Low-Rank Adaptation) provides an effective efficiency-accuracy trade-off for cross-domain few-shot learning with Vision Transformers.

### Research Questions
1. Does LoRA match or exceed full fine-tuning performance in cross-domain FSL?
2. What is the computational efficiency trade-off?
3. How does LoRA compare to Task-Specific Adapters?

## Repository Structure
```
bsc-dissertation-lora-fsl/
├── src/              # Source code (Python packages)
├── scripts/          # Training and evaluation scripts
├── configs/          # Experiment configurations
├── docs/             # Documentation and notes
├── notebooks/        # Analysis notebooks
├── data/             # → Symlink to scratch (datasets)
├── checkpoints/      # → Symlink to scratch (models)
├── logs/             # → Symlink to scratch (training logs)
└── results/          # Experimental results
    ├── tables/       # Summary tables (tracked in git)
    └── figures/      # Plots (tracked in git)
```

**Note:** `data/`, `checkpoints/`, and `logs/` are symlinks to scratch storage.

## Current Status

### Week 2 (February 2026)
- [x] Repository setup complete
- [x] PMF codebase running on Isambard
- [x] CIFAR-FS baseline: 85.9% (investigating gap from 92.5% reported)
- [ ] Reproduce miniImageNet baseline
- [ ] Reproduce Meta-Dataset baseline
- [ ] Implement LoRA adapters

## Setup

### On Isambard
```bash
# Clone repository (already done)
cd ~/bsc-dissertation-lora-fsl

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Datasets are in scratch via symlinks
# Download datasets to: ./data/raw/
```

## Baseline Reproduction

Target results from PMF paper:
- CIFAR-FS (5w5s): 92.5%
- miniImageNet (5w5s): 98.0%
- Meta-Dataset (avg): 83.1%

## Progress Tracking

See [docs/progress.md](docs/progress.md) for detailed weekly updates.

## References

1. Hu et al., "Pushing the Limits of Simple Pipelines for Few-Shot Learning", CVPR 2022
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
