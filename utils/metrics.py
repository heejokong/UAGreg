import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_open(pred, target, topk=(1,), num_classes=5):
    """Computes the precision@k for the specified values of k,
    num_classes are the number of known classes.
    This function returns overall accuracy,
    accuracy to reject unknown samples,
    the size of unknown samples in this batch."""
    maxk = max(topk)
    batch_size = target.size(0)
    pred = pred.view(-1, 1)
    pred = pred.t()
    ind = (target == num_classes)
    unknown_size = len(ind)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if ind.sum() > 0:
        unk_corr = pred.eq(target).view(-1)[ind]
        acc = torch.sum(unk_corr).item() / unk_corr.size(0)
    else:
        acc = 0

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], acc, unknown_size


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def roc_id_ood(score_id, score_ood):
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all)


def softmax_function(output):
    smax = to_np(F.softmax(output, dim=1))
    smax = np.max(smax, axis=1)
    return smax


def energy_function(output, T=1.):
    energy = to_np((T*torch.logsumexp(output / T, dim=1)))
    return energy

