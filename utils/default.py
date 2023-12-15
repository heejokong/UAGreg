import os
import math
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter

from datasets.ood import get_ood
from datasets.cifar import DATASET_GETTERS


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_dataset(args):
    labeled_dataset, unlabeled_dataset, test_dataset, val_dataset = DATASET_GETTERS[args.dataset](args)

    if args.dataset == 'cifar10':
        image_size = (32, 32, 3)
        ood_data = ["svhn", 'cifar100', 'lsun', 'imagenet']
    elif args.dataset == 'cifar100':
        image_size = (32, 32, 3)
        ood_data = ["svhn", 'cifar10', 'lsun', 'imagenet']
    elif args.dataset == 'imagenet':
        image_size = (224, 224, 3)
        # args.ood_data = ['lsun', 'dtd', 'cub', 'flowers102',
        #                  'caltech_256', 'stanford_dogs']
        ood_data = ["svhn", 'cifar10', 'cifar100', 'lsun']
    else:
        raise NotImplementedError()

    ood_loaders = {}
    for ood in ood_data:
        # ood_dataset = DATASET_GETTERS['ood'](args, ood, image_size=image_size, root='../data')
        ood_dataset = get_ood(ood, args.dataset, image_size=image_size)
        ood_loaders[ood] = torch.utils.data.DataLoader(
            ood_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    test_sampler = SequentialSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dataset,
        sampler=test_sampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    return labeled_loader, unlabeled_dataset, test_loader, val_loader, ood_loaders


def create_model(args):
    from models.wideresnet import wideresnet28
    from models.resnet import resnet18

    if args.arch == 'wideresnet':
        args.model_depth = 28
        args.model_width = 2
        model = wideresnet28(args.model_depth, args.model_width, args.num_classes)
    elif args.arch == 'resnet18':
        model = resnet18(num_classes=args.num_classes)
    else:
        raise NotImplementedError()

    return model


def set_optimizer(args, model):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # LOADING AN OPTIMIZER
    if args.optim == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=args.lr)
    else:
        raise NotImplementedError()

    # LOADING A SCHEDULER
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warm_up, args.total_step)

    return optimizer, scheduler


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class Logger(object):
    """ Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514 """
    def __init__(self, log_path, log_name='log.txt', local_rank=0, save_writer=False):
        self.local_rank = local_rank
        if self.local_rank in [-1, 0]:
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            self.path = log_path
            self.log_name = log_name
            open(os.path.join(self.path, self.log_name), 'w').close()
            if save_writer:
                self.writer = SummaryWriter(self.path)

    def info(self, msg):
        if self.local_rank in [-1, 0]:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(str(msg) + "\n")

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.local_rank in [-1, 0]:
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        if self.local_rank in [-1, 0]:
            self.writer.add_image(tag, images, step)

    def histo_summary(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        if self.local_rank in [-1, 0]:
            self.writer.add_histogram(tag, values, step, bins='auto')

