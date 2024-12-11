# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Main training function
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import utils
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
from models.pos_embed import interpolate_pos_embed
from models.head_downstream import PixelwiseTaskWithDPT

from stereoflow.datasets_stereo import get_train_dataset_stereo, get_test_datasets_stereo
from stereoflow.datasets_flow import get_train_dataset_flow, get_test_datasets_flow
from stereoflow.engine import train_one_epoch, validate_one_epoch
from stereoflow.criterion import *


def get_args_parser():
    # prepare subparsers 
    parser = argparse.ArgumentParser('Finetuning CroCo models on stereo or flow', add_help=False)
    subparsers = parser.add_subparsers(title="Task (stereo or flow)", dest="task", required=True)
    parser_stereo = subparsers.add_parser('stereo', help='Training stereo model')
    parser_flow = subparsers.add_parser('flow', help='Training flow model')
    def add_arg(name_or_flags, default=None, default_stereo=None, default_flow=None, **kwargs):
        if default is not None: assert default_stereo is None and default_flow is None, "setting default makes default_stereo and default_flow disabled"
        parser_stereo.add_argument(name_or_flags, default=default if default is not None else default_stereo, **kwargs)
        parser_flow.add_argument(name_or_flags, default=default if default is not None else default_flow, **kwargs)
    # output dir 
    add_arg('--output_dir', required=True, type=str, help='path where to save, if empty, automatically created')
    # model
    add_arg('--crop', type=int, nargs = '+', default_stereo=[352, 704], default_flow=[320, 384], help = "size of the random image crops used during training.")
    add_arg('--pretrained', required=True, type=str, help="Load pretrained model (required as croco arguments come from there)")
    # criterion  
    add_arg('--criterion', default_stereo='LaplacianLossBounded2()', default_flow='LaplacianLossBounded()', type=str, help='string to evaluate to get criterion')
    add_arg('--bestmetric', default_stereo='avgerr', default_flow='EPE', type=str)
    # dataset 
    add_arg('--dataset', type=str, required=True, help="training set")
    # training 
    add_arg('--seed', default=0, type=int, help='seed')
    add_arg('--batch_size', default_stereo=6, default_flow=8, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    add_arg('--epochs', default=32, type=int, help='number of training epochs')
    add_arg('--img_per_epoch', type=int, default=None, help='Fix the number of images seen in an epoch (None means use all training pairs)')
    add_arg('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    add_arg('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    add_arg('--lr', type=float, default_stereo=3e-5, default_flow=2e-5, metavar='LR', help='learning rate (absolute lr)')
    add_arg('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    add_arg('--warmup_epochs', type=int, default=1, metavar='N', help='epochs to warmup LR')
    add_arg('--optimizer', default='AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))', type=str,
                        help="Optimizer from torch.optim [ default: AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95)) ]")
    add_arg('--amp', default=0, type=int, choices=[0,1], help='enable automatic mixed precision training')
    # validation
    add_arg('--val_dataset', type=str, default='', help="Validation sets, multiple separated by + (empty string means that no validation is performed)")
    add_arg('--tile_conf_mode', type=str, default_stereo='conf_expsigmoid_15_3', default_flow='conf_expsigmoid_10_5', help='Weights for tile aggregation')
    add_arg('--val_overlap', default=0.7, type=float, help='Overlap value for the tiling')
    # others
    add_arg('--num_workers', default=8, type=int)
    add_arg('--eval_every', type=int, default=1, help='Val loss evaluation frequency')
    add_arg('--save_every', type=int, default=1, help='Save checkpoint frequency')
    add_arg('--start_from', type=str, default=None, help='Start training using weights from an other model (eg for finetuning)')
    add_arg('--tboard_log_step', type=int, default=100, help='Log to tboard every so many steps')
    add_arg('--dist_url', default='env://', help='url used to set up distributed training')

    return parser
    
        
def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    num_tasks = misc.get_world_size()

    assert os.path.isfile(args.pretrained)
    print("output_dir: "+args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Metrics / criterion 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    metrics = (StereoMetrics if args.task=='stereo' else FlowMetrics)().to(device)
    criterion = eval(args.criterion).to(device)
    print('Criterion: ', args.criterion)

    # Prepare model
    assert os.path.isfile(args.pretrained)
    ckpt = torch.load(args.pretrained, 'cpu')
    croco_args = croco_args_from_ckpt(ckpt)
    croco_args['img_size'] = (args.crop[0], args.crop[1])
    print('Croco args: '+str(croco_args))
    args.croco_args = croco_args # saved for test time 
    # prepare head 
    num_channels = {'stereo': 1, 'flow': 2}[args.task]
    if criterion.with_conf: num_channels += 1
    print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
    head = PixelwiseTaskWithDPT()
    head.num_channels = num_channels
    # build model and load pretrained weights
    model = CroCoDownstreamBinocular(head, **croco_args)
    interpolate_pos_embed(model, ckpt['model'])
    msg = model.load_state_dict(ckpt['model'], strict=False)
    print(msg)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Total params trainable: {total_params_trainable}")
    model_without_ddp = model.to(device)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print("lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], static_graph=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers   
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = eval(f"torch.optim.{args.optimizer}")
    print(optimizer)
    loss_scaler = NativeScaler()
    
    # automatic restart
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    if not args.resume and args.start_from:
        print(f"Starting from an other model's weights: {args.start_from}")
        best_so_far = None
        args.start_epoch = 0
        ckpt = torch.load(args.start_from, 'cpu')
        msg = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    else:
        best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if best_so_far is None: best_so_far = np.inf
    
    # tensorboard
    log_writer = None
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir, purge_step=args.start_epoch*1000)

    #  dataset and loader 
    print('Building Train Data loader for dataset: ', args.dataset)
    train_dataset = (get_train_dataset_stereo if args.task=='stereo' else get_train_dataset_flow)(args.dataset, crop_size=args.crop)
    def _print_repr_dataset(d):
        if isinstance(d, torch.utils.data.dataset.ConcatDataset):
            for dd in d.datasets:
                _print_repr_dataset(dd)
        else:
            print(repr(d))
    _print_repr_dataset(train_dataset)
    print('  total length:', len(train_dataset))
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if args.val_dataset=='':
        data_loaders_val = None
    else:
        print('Building Val Data loader for datasets: ', args.val_dataset)
        val_datasets = (get_test_datasets_stereo if args.task=='stereo' else get_test_datasets_flow)(args.val_dataset)
        for val_dataset in val_datasets: print(repr(val_dataset))
        data_loaders_val = [DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False) for val_dataset in val_datasets]
        bestmetric = ("AVG_" if len(data_loaders_val)>1 else str(data_loaders_val[0].dataset)+'_')+args.bestmetric
       
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # Training Loop
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed: data_loader_train.sampler.set_epoch(epoch)
            
        # Train
        epoch_start = time.time()
        train_stats = train_one_epoch(model, criterion, metrics, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)
        epoch_time = time.time() - epoch_start

        if args.distributed: dist.barrier()

        # Validation (current naive implementation runs the validation on every gpu ... not smart ...)
        if data_loaders_val is not None and args.eval_every > 0 and (epoch+1) % args.eval_every == 0:
            val_epoch_start = time.time()
            val_stats = validate_one_epoch(model, criterion, metrics, data_loaders_val, device, epoch, log_writer=log_writer, args=args)
            val_epoch_time = time.time() - val_epoch_start

            val_best = val_stats[bestmetric]
            
            # Save best of all
            if val_best <= best_so_far:
                best_so_far = val_best
                misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, best_so_far=best_so_far, fname='best')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         **{f'val_{k}': v for k, v in val_stats.items()}}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,}
                             
        if args.distributed: dist.barrier()
        
        # Save stuff
        if args.output_dir and ((epoch+1) % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, best_so_far=best_so_far, fname='last')

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)