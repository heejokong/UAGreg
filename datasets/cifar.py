import math
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from .randaugment import RandAugmentMC
from .imagenet import get_imagenet

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


def get_cifar10(args, root='../data/'):
    args.total_class = 10

    ### LOAD LABELED & UNLABELED DATASETS FOR TRAINING ###
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    base_dataset.targets = np.array(base_dataset.targets)
    if args.label_modify:
        base_dataset.targets -= 2
        base_dataset.targets[np.where(base_dataset.targets == -2)[0]] = 8
        base_dataset.targets[np.where(base_dataset.targets == -1)[0]] = 9
    labeled_idx, unlabeled_idx, val_idx = x_u_v_split(args, base_dataset.targets)

    ### DEFINE A TRANSFORMATION ###
    labeled_transform = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
    unlabeled_transform = TransformGraphMatch(mean=cifar10_mean, std=cifar10_std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

    labeled_dataset = CustomCIFAR10(args, root, labeled_idx, train=True, transform=labeled_transform)
    unlabeled_dataset = CustomCIFAR10(args, root, unlabeled_idx, train=True, transform=unlabeled_transform)
    val_dataset = CustomCIFAR10(args, root, val_idx, train=True, transform=test_transform)
    test_dataset = CustomCIFAR10(args, root, indexs=None, train=False, transform=test_transform)

    return labeled_dataset, unlabeled_dataset, test_dataset, val_dataset


def get_cifar100(args, root='../data/'):
    args.total_class = 100

    ### LOAD LABELED & UNLABELED DATASETS FOR TRAINING ###
    base_dataset = ModifiedCIFAR100(args, root, train=True, download=True)
    base_dataset.targets = np.array(base_dataset.targets)
    labeled_idx, unlabeled_idx, val_idx = x_u_v_split(args, base_dataset.targets)

    ### DEFINE A TRANSFORMATION ###
    labeled_transform = TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
    unlabeled_transform = TransformGraphMatch(mean=cifar10_mean, std=cifar10_std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])

    labeled_dataset = CustomCIFAR100(args, root, labeled_idx, train=True, transform=labeled_transform)
    unlabeled_dataset = CustomCIFAR100(args, root, unlabeled_idx, train=True, transform=unlabeled_transform)
    val_dataset = CustomCIFAR100(args, root, val_idx, train=True, transform=test_transform)
    test_dataset = CustomCIFAR100(args, root, indexs=None, train=False, transform=test_transform)

    return labeled_dataset, unlabeled_dataset, test_dataset, val_dataset


def x_u_v_split(args, labels):
    label_per_class = args.num_labeled #// args.num_classes
    val_per_class = args.num_val #// args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    val_idx = []
    unlabeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        unlabeled_idx.extend(idx)
        idx = np.random.choice(idx, label_per_class+val_per_class, False)
        labeled_idx.extend(idx[:label_per_class])
        val_idx.extend(idx[label_per_class:])

    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled * args.num_classes
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    #if not args.no_out:
    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in labeled_idx]
    unlabeled_idx = [idx for idx in unlabeled_idx if idx not in val_idx]
    return labeled_idx, unlabeled_idx, val_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))


class TransformOpenMatch(object):
    def __init__(self, mean, std, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))


class TransformGraphMatch(object):
    def __init__(self, mean, std, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.strong_1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.strong_2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong_1 = self.strong_1(x)
        strong_2 = self.strong_2(x)
        return self.normalize(weak), self.normalize(strong_1), self.normalize(strong_2)


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, args, root, indexs=None, train=True, transform=None, target_transform=None, download=False, return_idx=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = np.array(self.targets)

        if args.label_modify:
            self.targets -= 2
            self.targets[np.where(self.targets == -2)[0]] = 8
            self.targets[np.where(self.targets == -1)[0]] = 9

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.return_idx = return_idx
        self.set_index()


    def set_index(self, indexs=None):
        if indexs is not None:
            self.data_index = self.data[indexs]
            self.targets_index = self.targets[indexs]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, idx):
        img, target = self.data_index[idx], self.targets_index[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, idx
        return img, target

    def __len__(self):
        return len(self.data_index)


class ModifiedCIFAR100(datasets.CIFAR100):
    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        hier_labels = np.array([
            [ 4, 30, 55, 72, 95],
            [ 1, 32, 67, 73, 91],
            [54, 62, 70, 82, 92],
            [ 9, 10, 16, 28, 61],
            [ 0, 51, 53, 57, 83],
            [22, 39, 40, 86, 87],
            [ 5, 20, 25, 84, 94],
            [ 6,  7, 14, 18, 24],
            [ 3, 42, 43, 88, 97],
            [12, 17, 37, 68, 76],
            [23, 33, 49, 60, 71],
            [15, 19, 21, 31, 38],
            [34, 63, 64, 66, 75],
            [26, 45, 77, 79, 99],
            [ 2, 11, 35, 46, 98],
            [27, 29, 44, 78, 93],
            [36, 50, 65, 74, 80],
            [47, 52, 56, 59, 96],
            [ 8, 13, 48, 58, 90],
            [41, 69, 81, 85, 89],
            ])
        assert args.num_classes % 20 == 0

        self.targets = np.array(self.targets)
        if args.label_biased:
            per_coarse = args.num_classes // 20
            known_classes = np.unique(hier_labels[:,:per_coarse].flatten())
            unknown_classes = np.unique(hier_labels[:,per_coarse:].flatten())
        else:
            num_coarse = args.num_classes // 5
            known_classes = np.unique(hier_labels[:num_coarse,:].flatten())
            unknown_classes = np.unique(hier_labels[num_coarse:,:].flatten())

        self.targets_new = np.zeros_like(self.targets)
        for i, known in enumerate(known_classes):
            ind_known = np.where(self.targets==known)[0]
            self.targets_new[ind_known] = i
        for i, unknown in enumerate(unknown_classes):
            ind_unknown = np.where(self.targets == unknown)[0]
            self.targets_new[ind_unknown] = args.num_classes
        self.targets = self.targets_new


class CustomCIFAR100(ModifiedCIFAR100):
    def __init__(self, args, root, indexs=None, train=True, transform=None, target_transform=None, download=False, return_idx=False):
        super().__init__(args, root, train=train, transform=transform, target_transform=target_transform, download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.return_idx = return_idx
        self.set_index()

    def set_index(self, indexs=None):
        if indexs is not None:
            self.data_index = self.data[indexs]
            self.targets_index = self.targets[indexs]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, idx):
        img, target = self.data_index[idx], self.targets_index[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_idx:
            return img, target, idx
        return img, target

    def __len__(self):
        return len(self.data_index)


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'imagenet': get_imagenet,
                   }

