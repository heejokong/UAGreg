This is an PyTorch implementation of UAG.
This implementation is based on [Pytorch-FixMatch](https://github.com/kekmodel/FixMatch-pytorch).

## Usage

### Dataset Preparation

This repository needs CIFAR-10, CIFAR-100, or ImageNet-30 to train a model.

To fully reproduce the results in evaluation, we also need SVHN, LSUN, ImageNet
for CIFAR10, 100, and LSUN, DTD, CUB, Flowers, Caltech_256, Stanford Dogs for ImageNet-30.
To prepare the datasets above, follow [CSI](https://github.com/alinlab/CSI), [OpenMatch](https://github.com/VisionLearningGroup/OP_Match).

All datasets are supposed to be under ./data.

### Train

Train the model for CIFAR-10 dataset under correlated and uncorrelated settings:

```
sh run_cifar10_corr.sh (gpu_id) (num_labeled)
sh run_cifar10_uncorr.sh (gpu_id) (num_labeled)
```

Train the model for CIFAR-100 dataset under correlated and uncorrelated settings:

```
sh run_cifar100_corr.sh (gpu_id) (num_labeled)
sh run_cifar100_uncorr.sh (gpu_id) (num_labeled)
```

Run experiments on ImageNet-30:

```
sh run_imagenet.sh (gpu_id)
```
