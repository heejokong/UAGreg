import os
import copy
import math
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from utils.misc import AverageMeter
from utils.metrics import accuracy, accuracy_open, compute_roc, roc_id_ood
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from datasets.cifar import TransformOpenMatch
from datasets.imagenet import TransformOpenMatch_Imagenet

best_acc = 0.
best_acc_val = 0.

def train_loop(
    args, model, ema_model, optimizer, scheduler, 
    labeled_loader, unlabeled_dataset, test_loader, val_loader, ood_loaders, 
    logger=None):
    
    if args.use_amp:
        from apex import amp

    global best_acc
    global best_acc_val

    end = time.time()

    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        size_image = 32
        trans_func = TransformOpenMatch
        all_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image, padding=int(size_image*0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size_image = 32
        trans_func = TransformOpenMatch
        all_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image, padding=int(size_image*0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size_image = 224
        trans_func = TransformOpenMatch_Imagenet
        all_transform = transforms.Compose([
            transforms.RandomResizedCrop(size_image, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
            ])

    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    unlabeled_dataset_all.transform = all_transform

    labeled_dataset = copy.deepcopy(labeled_loader.dataset)
    labeled_dataset.transform = trans_func(mean=mean, std=std)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # 
    labeled_loader = DataLoader(
        labeled_dataset, sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True
        )
    #
    unlabeled_loader = DataLoader(
        unlabeled_dataset, sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu, num_workers=args.num_workers, drop_last=True
        )
    # 
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # 
    out_th = 1.0
    lower_th = 1.0
    upper_th = 1.0

    ## TRAINING LOOP ##
    for epoch in range(args.start_epoch, args.epochs):
        logger.info("\nEPOCH {:4}/{:4}".format(epoch+1, args.epochs))
        # 
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter() # TOTAL LOSS
        # SEMI-SUPERVISED LEARNING LOSS
        losses_s = AverageMeter()
        losses_u = AverageMeter()
        # PENALIZING LOSS FOR OUT-OF-DISTRIBUTION SAMPLES
        losses_o = AverageMeter()
        # GRAPH-REGULARIZATION LOSS
        losses_g = AverageMeter()
        mask_probs = AverageMeter()

        if epoch > 0:
            curr_lower_th, curr_upper_th = get_threshold(args, unlabeled_dataset_all, model)
            lower_th = args.momentum * lower_th + (1. - args.momentum) * curr_lower_th
            upper_th = args.momentum * upper_th + (1. - args.momentum) * curr_upper_th
            out_th = [lower_th, upper_th]
            # 
            # print("THRESHOLD ({:4}/{:4}): {}".format(epoch+1, args.epochs, out_th))
        logger.info("THRESHOLD ({:4}/{:4}): {}".format(epoch+1, args.epochs, out_th))
        p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()
        for b_idx in p_bar:
            try:
                (_, inputs_x_s, inputs_x), targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_loader)
                (_, inputs_x_s, inputs_x), targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s1, inputs_u_s2), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s1, inputs_u_s2), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            bl_size = inputs_x.shape[0]
            bu_size = inputs_u_w.shape[0]

            inputs_all = torch.cat([inputs_u_w, inputs_u_s1, inputs_u_s2], 0)
            inputs = torch.cat([inputs_x, inputs_x_s, inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)

            ## FORWARD PROPAGATION ##
            outputs, outputs_open, feats = model(inputs)
            feats_u_w, feats_u_s1, feats_u_s2 = feats[2*bl_size:].chunk(3)
            outputs_u_w, outputs_u_s1, outputs_u_s2 = outputs[2*bl_size:].chunk(3)
            outputs_open_u_w, outputs_open_u_s1, outputs_open_u_s2 = outputs_open[2*bl_size:].chunk(3)

            ## SUPERIVSED LOSS FOR LABELED SAMPLES ##
            loss_s = F.cross_entropy(outputs[:2*bl_size], targets_x.repeat(2), reduction='mean')
            loss_s += F.cross_entropy(outputs_open[:2*bl_size], targets_x.repeat(2), reduction='mean')

            """ CONTRIBUTION_1. """
            ## PENALIZING LOSS FOR OUTLIER SAMPLES ##
            ## MAXIMUM SOFTMAX PROBABILITY
            ood_score, _ = torch.max(torch.softmax(outputs_open_u_w / args.temp_o, dim=1), dim=-1)
            # 
            id_mask = ood_score.ge(upper_th)
            ood_mask = ood_score.lt(lower_th)
            graph_mask = ood_mask.clone()
            #
            me_max = True
            if me_max:
                probs = torch.softmax(outputs_open_u_w, dim=1)
                avg_probs = (probs[ood_mask]).mean(dim=0)
                loss_o = -torch.sum(-avg_probs * torch.log(avg_probs + 1e-8))
            else:
                loss_o = torch.zeros(1).to(args.device).mean()

            """ CONTRIBUTION_2. """
            ## GRAPH REGULARIZATION LOSS FOR UNLABELED SAMPLES ##
            graph_reg = True
            if graph_reg:
                with torch.no_grad():
                    outputs_u_w = outputs_u_w.detach()
                    probs = torch.softmax(outputs_u_w, dim=1)

                # EMBEDDING SIMILARITY
                sim = torch.exp(torch.mm(feats_u_s1, feats_u_s2.t()) / args.T) 
                sim_probs = sim / sim.sum(1, keepdim=True)

                # PSEUDO-LABEL GRAPH
                Q = torch.mm(probs, probs.t())
                Q.fill_diagonal_(1)
                pos_mask = (Q >= args.graph_th).float()
                graph_mask = (graph_mask.unsqueeze(0) == graph_mask.unsqueeze(-1)).float()

                Q = Q * pos_mask
                Q = Q * graph_mask
                Q = Q / Q.sum(1, keepdim=True)

                # CONTRASTIVE LOSS
                loss_g = -(torch.log(sim_probs + 1e-7) * Q).sum(1)
                loss_g = loss_g.mean()
            else:
                loss_g = torch.zeros(1).to(args.device).mean()

            ## UNSUPERIVSED LOSS FOR UNLABELED SAMPLES ##
            if epoch >= args.start_u:
                pseudo_label = torch.softmax(outputs_u_w.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.conf_th).float()
                mask = mask * id_mask.float()
                loss_u = (F.cross_entropy(outputs_u_s1, targets_u, reduction='none') * mask).mean()
                mask_probs.update(mask.mean().item())
            else:
                loss_u = torch.zeros(1).to(args.device).mean()

            loss = loss_s + args.lambda_u * loss_u + args.lambda_g * loss_g
            loss += loss_o

            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_s.update(loss_s.item())
            losses_u.update(loss_u.item())
            losses_g.update(loss_g.item())

            optimizer.step()
            if args.optim != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            p_bar.set_description(
                "TRAIN EPOCH: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. L_S: {loss_s:.4f}. L_U:{loss_u:.4f}. L_G:{loss_g:.4f}. MASK: {mask:.2f}.".format(
                    epoch=epoch+1,
                    epochs=args.epochs,
                    batch=b_idx+1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    loss_s=losses_s.average,
                    loss_u=losses_u.average,
                    loss_g=losses_g.average,
                    mask=mask_probs.average
                ))

        if args.local_rank in [-1, 0]:
            p_bar.close()

        ## EVALUATION ##
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            val_loss, val_acc = eval_val(args, val_loader, test_model)
            test_loss, test_acc, test_roc_sm, test_roc_energy \
                    = eval_test(args, test_loader, test_model)

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                best_acc_val = val_acc
                best_acc = test_acc
                best_model = copy.deepcopy(test_model)

            logger.info("EVALUATION RESULTS AT {:4}/{:4}".format(epoch+1, args.epochs))
            logger.info("VAL ACC: {:.4f}".format(val_acc))
            logger.info("TEST ACC: {:.4f}".format(test_acc))
            logger.info("TEST AUROC (MSP): {:.4f}".format(test_roc_sm))
            logger.info("TEST AUROC (Energy): {:.4f}".format(test_roc_energy))

            if (epoch + 1) % 10 == 0:
                ### SAVING THE MODEL ###
                if args.local_rank in [-1, 0]:
                    model_to_save = model.module if hasattr(model, "module") else model
                    state = {
                        'state_dict': model_to_save.state_dict(),
                        'best_acc_val': best_acc_val,
                        'best_acc': best_acc,
                    }
                    if args.use_ema:
                        ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
                        state['ema_state_dict'] = ema_to_save.state_dict()
                    filepath = os.path.join(args.out, f'model_checkpoint_{epoch+1}.pth')
                    torch.save(state, filepath)

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


def get_threshold(args, dataset, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    dataloader = tqdm(dataloader, disable=args.local_rank not in [-1, 0])

    pred_scores = torch.Tensor([]).to(args.device)
    pred_labels = torch.Tensor([]).to(args.device)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, _) in enumerate(dataloader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open, _ = model(inputs)
            max_probs, preds = torch.max(F.softmax(outputs_open / args.temp_o, dim=1), dim=1)

            # 
            pred_scores = torch.cat((pred_scores, max_probs), dim=0)
            pred_labels = torch.cat((pred_labels, preds), dim=0)

            batch_time.update(time.time() - end)
            end = time.time()

            dataloader.set_description(
                "PRE ITER: {batch:4}/{iter:4}. DATA_TIME: {dt:.3f}sec. BATCH_TIME: {bt:.3f}sec.".format(
                    batch=batch_idx+1,
                    iter=len(dataloader),
                    dt=data_time.average,
                    bt=batch_time.average
                ))
        dataloader.close()

    # FOR GAUSSIAN MIXTURE MODELS (GMMs)
    otsu_th = threshold_otsu(pred_scores.cpu().numpy())
    init_centers = np.array([[otsu_th], [otsu_th]])
    gmm = GaussianMixture(n_components=2, means_init=init_centers)
    # 
    gmm.fit(pred_scores.unsqueeze(-1).cpu().numpy())
    threshold = np.squeeze(gmm.means_, axis=1)
    threshold.sort()

    return threshold

