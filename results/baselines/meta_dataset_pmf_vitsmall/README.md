# Meta-Dataset Baseline (PMF, DINO/IN1K, ViT-small)

**Result (Avg over 10 domains): 83.06%**  
Paper reference: 83.13% (P>M>F DINO/IN1K ViT-small)

## Run info
- Date: 2026-02-11
- Slurm job id: 2275716
- Node: nid010510
- GPUs: 4x NVIDIA GH200 120GB
- Conda env: pmf (Python 3.10.13, Torch 2.10.0+cu128)
- PMF repo: /scratch/u5hv/frixos.u5hv/pmf_cvpr22
- Data: /scratch/u5hv/frixos.u5hv/data/meta_dataset_h5
- Checkpoint: /scratch/u5hv/frixos.u5hv/pmf_metadataset_dino/md_full_128x128_dinosmall_fp16_lr5e-5/best.pth
- Eval mode: finetune (deploy=finetune), LR selected per domain (see log table)
- Protocol: fixed 5-way 5-shot (test_n_way=5, n_shot=5)

## Per-domain (Acc@1)
traffic_sign 90.07
mscoco       59.54
ilsvrc_2012  73.01
omniglot     92.00
aircraft     89.55
cu_birds     91.05
dtd          85.64
quickdraw    79.36
fungi        75.17
vgg_flower   95.19
