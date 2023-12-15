import os
import math
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms


DATA_PATH = '../data'

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_transform(mean, std, image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_ood(dataset, id, test_only=False, image_size=None):
    image_size = (32, 32, 3) if image_size is None else image_size
    if id == "cifar10":
        mean = cifar10_mean
        std = cifar10_std
    elif id == "cifar100":
        mean = cifar100_mean
        std = cifar100_std
    elif "imagenet"  in id or id == "tiny":
        mean = normal_mean
        std = normal_std

    _, test_transform = get_transform(mean, std, image_size=image_size)

    if dataset == 'cifar10':
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=False,
                                    transform=test_transform)

    elif dataset == 'cifar100':
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=False,
                                     transform=test_transform)

    elif dataset == 'svhn':
        test_set = datasets.SVHN(DATA_PATH, split='test', download=True,
                                 transform=test_transform)

    elif dataset == 'lsun':
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    elif dataset == 'stanford_dogs':
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'cub':
        test_dir = os.path.join(DATA_PATH, 'cub')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'flowers102':
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'food_101':
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'caltech_256':
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'dtd':
        test_dir = os.path.join(DATA_PATH, 'dtd')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'pets':
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    return test_set
