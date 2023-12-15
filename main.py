import os
import math
import numpy as np
import torch

from utils.default import set_seed, set_dataset, create_model, set_optimizer, Logger
from utils.parser import get_parser
from models.ema import ModelEMA
from trainer import train_loop


def main():
    args = get_parser()

    ### SET TORCH DEVICE ###
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpus = 1
    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    ### LOAD TRAIN & TEST DATASETS ###
    labeled_loader, unlabeled_dataset, test_loader, val_loader, ood_loaders = set_dataset(args)

    ### LOAD AN INITIALIZED MODEL ###
    model = create_model(args)
    model.to(args.device)

    ema_model = None
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    ### DEFINE AN OPTIMIZER WITH SCHEDULER ###
    args.epochs = math.ceil(args.total_step / args.eval_step)
    optimizer, scheduler = set_optimizer(args, model)

    ### LOAD DISTRIBUTED_DATA_PARALLE(DDP) WITH NVIDIA APEX ###
    if args.use_amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    ### TRAINING A MODEL ###
    ln = f"{args.dataset}_{args.trainer}_{args.arch}.txt"
    logger = Logger(log_path=args.out, log_name=ln, local_rank=args.local_rank)
    logger.info(args)
    logger.info(f"***** TRAINING A {args.arch.upper()} MODEL FOR {args.dataset.upper()}*****")
    logger.info("TOTAL MODEL PARAMS: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    ### TRAINING LOOP ###
    best_model, best_acc = train_loop(
        args, model, ema_model, optimizer, scheduler, 
        labeled_loader, unlabeled_dataset, test_loader, val_loader, ood_loaders,
        logger)

    ### SAVING THE MODEL ###
    if args.local_rank in [-1, 0]:
        model_to_save = best_model.module if hasattr(model, "module") else best_model
        state = {'state_dict': model_to_save.state_dict(),
                 'best_acc': best_acc,
                 }
        filepath = os.path.join(args.out, 'model_best.pth')
        torch.save(state, filepath)


if __name__ == "__main__":
    main()

