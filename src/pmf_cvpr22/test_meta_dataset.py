import os
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json

from copy import deepcopy
from pathlib import Path
from tabulate import tabulate

from engine import evaluate
import utils.deit_util as utils
from datasets import get_sets
from utils.args import get_args_parser
from models import get_model
from models.lora import inject_lora, eject_lora
from datasets import get_loaders


def get_test_loader(args):
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if args.distributed:
        _, data_loader_val = get_loaders(args, num_tasks, global_rank)
    else:
        _, _, dataset_val = get_sets(args)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            generator=generator
        )
    return data_loader_val


def main(args):
    utils.init_distributed_mode(args)

    args.eval = True
    args.dataset = 'meta_dataset'

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    ##############################################
    # Model
    print(f"Creating model: {args.deploy} {args.arch}")

    model = get_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.unused_params)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print(f'Load ckpt from {args.resume}')

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    ##############################################
    # Test
    criterion = torch.nn.CrossEntropyLoss()
    #datasets = ['mscoco', 'traffic_sign', 'ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
    datasets = args.test_sources
    var_accs = {}

    for domain in datasets:
        print(f'\n# Testing {domain} starts...\n')

        args.test_sources = [domain]
        data_loader_val = get_test_loader(args)

        # validate lr
        best_lr = args.ada_lr
        best_r = getattr(args, 'lora_r', None)
        if args.deploy in ['finetune', 'finetune_lora']:
            print("Start selecting the best lr...")
            best_acc = 0
            for lr in [0, 0.0001, 0.001, 0.01]:
                model_without_ddp.lr = lr
                test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5)
                acc = test_stats['acc1']
                print(f"*lr = {lr}: acc1 = {acc}")
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            model_without_ddp.lr = best_lr
            print(f"### Selected lr = {best_lr}")

        elif args.deploy == 'finetune_lora_adaptive':
            print("Start adaptive LoRA search (r x lr)...")
            best_acc = 0
            for r in [1, 2, 4, 8, 16, 32, 64]:
                torch.manual_seed(1234)
                np.random.seed(1234)
                random.seed(1234)
                model_without_ddp.backbone.load_state_dict(model_without_ddp.backbone_state, strict=True)
                eject_lora(model_without_ddp.backbone)
                inject_lora(model_without_ddp.backbone, r, r, targets=args.lora_target)
                model_without_ddp.backbone_state = deepcopy(model_without_ddp.backbone.state_dict())

                for lr in [0, 0.0001, 0.001, 0.01]:
                    torch.manual_seed(1234)
                    np.random.seed(1234)
                    random.seed(1234)
                    model_without_ddp.lr = lr
                    test_stats = evaluate(data_loader_val, model, criterion, device, seed=1234, ep=5)
                    acc = test_stats['acc1']
                    print(f"*r={r}, lr={lr}: acc1={acc}")
                    if acc > best_acc:
                        best_acc = acc
                        best_lr = lr
                        best_r = r
            # set best config for final eval
            torch.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)

            model_without_ddp.backbone.load_state_dict(model_without_ddp.backbone_state, strict=True)
            eject_lora(model_without_ddp.backbone)
            inject_lora(model_without_ddp.backbone, best_r, best_r, targets=args.lora_target)
            model_without_ddp.backbone_state = deepcopy(model_without_ddp.backbone.state_dict())
            model_without_ddp.lr = best_lr
            print(f"### Selected r={best_r}, lr={best_lr}") 


        # final classification
        data_loader_val.generator.manual_seed(args.seed + 10000)
        test_stats = evaluate(data_loader_val, model, criterion, device)
        var_accs[domain] = (test_stats['acc1'], test_stats['acc_std'], best_lr, best_r)

        print(f"{domain}: acc1 on {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")

        if args.output_dir and utils.is_main_process():
            test_stats['domain'] = args.test_sources[0]
            test_stats['lr'] = best_lr
            test_stats['lora_r'] = best_r
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(json.dumps(test_stats) + "\n")

    # print results as a table
    if utils.is_main_process():
        rows = []
        for dataset_name in datasets:
            row = [dataset_name]
            acc, std, lr, r = var_accs[dataset_name]
            conf = (1.96 * std) / np.sqrt(len(data_loader_val.dataset))
            row.append(f"{acc:0.2f} +- {conf:0.2f}")
            row.append(f"{lr}")
            row.append(f"{r}" if r is not None else "-")
            rows.append(row)
        np.save(os.path.join(output_dir, f'test_results_{args.deploy}_{args.train_tag}.npy'), {'rows': rows})

        table = tabulate(rows, headers=['Domain', args.arch, 'lr', 'lora_r'], floatfmt=".2f")
        print(table)
        print("\n")

        import tables
        tables.file._open_files.close_all()

        if args.output_dir:
            with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
                f.write(table)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.train_tag = 'pt' if args.resume == '' else 'ep'
    args.train_tag += f'_step{args.ada_steps}_lr{args.ada_lr}_prob{args.aug_prob}'

    if utils.is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import sys
        with (output_dir / f"log_test_{args.deploy}_{args.train_tag}.txt").open("a") as f:
            f.write(" ".join(sys.argv) + "\n")

    main(args)
