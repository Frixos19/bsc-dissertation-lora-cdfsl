# PMF Baseline Reproduction

Backbone: DINO-small (ViT-S/16)
Architecture: dino_small_patch16
Epochs: 100
LR: 5e-5
Episodes (eval): 600
FP16: True

Results:

CIFAR-FS 5w1s: 82.02 ± 0.81
CIFAR-FS 5w5s: 92.44 ± 0.49
miniImageNet 5w1s: 93.26 ± 0.47
miniImageNet 5w5s: 98.17 ± 0.16

All results closely match PMF CVPR 2022.

