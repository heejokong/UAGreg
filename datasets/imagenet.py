import math
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from .randaugment import RandAugmentMC
from .mydataset import ImageFolder, ImageFolder_fix


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_imagenet(args, norm=True):
    mean = normal_mean
    std = normal_std
    txt_labeled = "filelist/imagenet_train_labeled.txt"
    txt_unlabeled = "filelist/imagenet_train_unlabeled.txt"
    txt_val = "filelist/imagenet_val.txt"
    txt_test = "filelist/imagenet_test.txt"

    ## This function will be overwritten in trainer.py
    norm_func = TransformGraphMatch_Imagenet(mean=mean, std=std,
                                           norm=norm, size_image=224)
    labeled_dataset = ImageFolder(txt_labeled, transform=norm_func)
    unlabeled_dataset = ImageFolder_fix(txt_unlabeled, transform=norm_func)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_dataset = ImageFolder(txt_val, transform=test_transform)
    test_dataset = ImageFolder(txt_test, transform=test_transform)

    return labeled_dataset, unlabeled_dataset, test_dataset, val_dataset


class TransformGraphMatch_Imagenet(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size_image, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size_image, scale=(0.2, 1.)),                
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),                         
            transforms.RandomHorizontalFlip(),
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong_1 = self.strong(x)
        strong_2 = self.strong(x)
        return self.normalize(weak), self.normalize(strong_1), self.normalize(strong_2)


class TransformOpenMatch_Imagenet(object):
    def __init__(self, mean, std, norm=True, size_image=224):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=size_image),
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak2(x)
        weak2 = self.weak2(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(weak2)
        else:
            return weak, strong

