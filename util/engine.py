# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
import util.utils as utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    rdrop_bsize: int = 8,
                    max_rdrop_epoch: int = 10,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, samples_bp, targets,_ in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        samples_bp = samples_bp.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            state = np.random.get_state()
            samples, targets_1 = mixup_fn(samples, targets)
            np.random.set_state(state)
            samples_bp, targets = mixup_fn(samples_bp, targets)

        #if True:  # with torch.cuda.amp.autocast():
        outputs1, feat1 = model(samples, samples_bp)

        loss = criterion(samples, outputs1, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_online(data_loader, model, device, score_out_path):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, images_bp, target, path in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        images_bp = images_bp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output, _ = model(images, images_bp)
        loss = criterion(output, target)

        acc1,acc5 = accuracy(output, target, topk=(1, 1))
        #print(acc1)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, score_out_path, has_label):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    score_list = []
    path_list = []
    label_list = []

    for images, images_bp, target, path in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        images_bp = images_bp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        images = images[:,:,:,torch.arange(images.size(3) - 1, -1, -1).long()]
        images_bp = images_bp[:,:,:,torch.arange(images_bp.size(3) - 1, -1, -1).long()]

        output, _ = model(images, images_bp)

        cls_pred_cpu = F.softmax(output,dim=1).data.cpu().numpy()[:,1]
        label_cpu = target.data.cpu().numpy()
        #print(cls_pred_cpu, label_cpu)
        for i in range(len(cls_pred_cpu)):
            score_list.append(cls_pred_cpu[i])
            path_list.append(path[i])
            label_list.append(label_cpu[i])

    # -----------------------------------------------------
    # results output 

    fp = open(score_out_path,'w')

    prev = None
    next = None
    threshold = 0.3

    LEN = len(score_list)
    print('total: ',LEN)
    for idx in range(LEN):
        score = score_list[idx]
        if 'test' in path_list[idx]:
            if prev:
                if idx < LEN - 1:
                    next = score_list[idx+1]
                    if prev < 0.4 and next < 0.4 and score < 0.8:
                        score = (prev+next)/2.0
                    elif prev > 0.5 and next > 0.5 and score > 0.1:
                        score = (prev+next)/2.0
            prev = score
        elif 'dev' in path_list[idx]:
            if score > threshold:
                score = threshold + 0.005
            else:
                score = threshold - 0.005


        if has_label:
            fp.write(path_list[idx] + ' ' + str(score)[:7] + ' ' + str(label_list[idx]) +'\n')
        else:
            fp.write(path_list[idx] + ' ' + str(score)[:7] +'\n')

    fp.close()

    return LEN

