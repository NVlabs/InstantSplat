# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Pre-training CroCo 
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math 
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models.croco import CroCoNet
from models.criterion import MaskedMSE
from datasets.pairs_dataset import PairsDataset


def get_args_parser():
    parser = argparse.ArgumentParser('CroCo pre-training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default='CroCoNet()', type=str, help="string containing the model to build")
    parser.add_argument('--norm_pix_loss', default=1, choices=[0,1], help="apply per-patch mean/std normalization before applying the loss")
    # dataset 
    parser.add_argument('--dataset', default='habitat_release', type=str, help="training set")
    parser.add_argument('--transforms', default='crop224+acolor', type=str, help="transforms to apply") # in the paper, we also use some homography and rotation, but find later that they were not useful or even harmful
    # training 
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--max_epoch', default=400, type=int, help="Stop training at this epoch")
    parser.add_argument('--accum_iter', default=1, type=int, help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')              
    parser.add_argument('--amp', type=int, default=1, choices=[0,1], help="Use Automatic Mixed Precision for pretraining")
    # others 
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int, help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int, help='frequence (number of iterations) to print infos while training')
    # paths 
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    parser.add_argument('--data_dir', default='./data/', type=str, help="path where data are stored")
    return parser



        
def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    
    print("output_dir: "+args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)                         

    # auto resume 
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ## training dataset and loader 
    print('Building dataset for {:s} with transforms {:s}'.format(args.dataset, args.transforms))
    dataset = PairsDataset(args.dataset, trfs=args.transforms, data_dir=args.data_dir)
    if world_size>1:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)
    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
   
    ## model 
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print('Loading criterion: MaskedMSE(norm_pix_loss={:s})'.format(str(bool(args.norm_pix_loss))))
    criterion = MaskedMSE(norm_pix_loss=bool(args.norm_pix_loss))
   
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module
    
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay) # following timm: set wd as 0 for bias and norm layers
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training until {args.max_epoch} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.max_epoch):
        if world_size>1:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and epoch % args.save_freq == 0 :
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, fname='last')
                
        if args.output_dir and (epoch % args.keep_freq == 0 or epoch + 1 == args.max_epoch) and (epoch>0 or args.max_epoch==1):
            misc.save_model(
                args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (image1, image2) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        # we use a per iteration  lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        image1 = image1.to(device, non_blocking=True) 
        image2 = image2.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            out, mask, target = model(image1, image2)
            loss = criterion(out, mask, target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and ((data_iter_step + 1) % (accum_iter*args.print_freq)) == 0:
            # x-axis is based on epoch_1000x in the tensorboard, calibrating differences curves when batch size changes 
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
