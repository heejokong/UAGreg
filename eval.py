import os
import copy
import math
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import AverageMeter
from utils.metrics import accuracy, accuracy_open, compute_roc, roc_id_ood

best_acc = 0.
best_acc_val = 0.

def eval_loop(
    args, model, ema_model, optimizer, scheduler, 
    test_loader, val_loader, ood_loaders, 
    logger=None):
    
    if args.use_amp:
        from apex import amp

    global best_acc
    global best_acc_val

    if args.local_rank in [-1, 0]:
        val_loss, val_acc = eval_val(args, val_loader, model)
        test_loss, test_acc, test_roc_sm, test_roc_energy = eval_test(args, test_loader, model)

        logger.info("VAL ACC: {:.4f}".format(val_acc))
        logger.info("TEST ACC: {:.4f}".format(test_acc))
        logger.info("TEST AUROC (MSP): {:.4f}".format(test_roc_sm))
        logger.info("TEST AUROC (Energy): {:.4f}".format(test_roc_energy))

    if args.local_rank in [-1, 0]:
        return best_model, best_acc
    else:
        return None, None


def eval_val(args, testloader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    testloader = tqdm(testloader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs, _, _ = model(inputs)
            outputs = F.softmax(outputs, 1)

            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1)) #[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, _ = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()

            testloader.set_description(
                "TEST ITER: {batch:4}/{iter:4}. DATA_TIME: {dt:.3f}sec. BATCH_TIME: {bt:.3f}sec.".format(
                    batch=batch_idx+1,
                    iter=len(testloader),
                    dt=data_time.average,
                    bt=batch_time.average
                ))
        testloader.close()

    return losses.average, top1.average


def eval_test(args, testloader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # K-WAY CLASSIFICATION
    losses = AverageMeter()
    top1 = AverageMeter()
    # OUTLIER DETECTION
    accs_all = AverageMeter()
    accs_unk = AverageMeter()
    end = time.time()

    testloader = tqdm(testloader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs, outputs_open, _ = model(inputs)

            ## GET SOFTMAX SCORE
            outputs_known = F.softmax(outputs_open / args.temp_o, dim=1)
            known_score, pred_close = outputs_known.data.max(dim=1)
            ## GET ENERGY SCORE
            temp_e = 1.0
            energy_score = temp_e * torch.logsumexp(outputs_open / temp_e, dim=1)

            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1))
            known_targets = targets < int(outputs.size(1)) #[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, _ = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                known_all = known_score
                label_all = targets
                energy_all = energy_score
            else:
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)
                energy_all = torch.cat([energy_all, energy_score], 0)

            testloader.set_description(
                "TEST ITER: {batch:4}/{iter:4}. DATA_TIME: {dt:.3f}sec. BATCH_TIME: {bt:.3f}sec.".format(
                    batch=batch_idx+1,
                    iter=len(testloader),
                    dt=data_time.average,
                    bt=batch_time.average
                ))
        testloader.close()

    ## ROC calculation
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    roc_soft = compute_roc(-known_all, label_all, num_known=int(outputs.size(1)))

    energy_all = energy_all.cpu().numpy()
    roc_energy = compute_roc(-energy_all, label_all, num_known=int(outputs.size(1)))

    ind_known = np.where(label_all < int(outputs.size(1)))[0]
    id_score = known_all[ind_known]

    return losses.average, top1.average, \
           roc_soft, roc_energy
