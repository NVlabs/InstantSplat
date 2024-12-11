# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Main function for training one epoch or testing
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import numpy as np
import torch
import torchvision

from utils import misc as misc


def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, metrics: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, print_freq = 20,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    details = {}

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if args.img_per_epoch:
        iter_per_epoch = args.img_per_epoch // args.batch_size + int(args.img_per_epoch % args.batch_size > 0)
        assert len(data_loader) >= iter_per_epoch, 'Dataset is too small for so many iterations'
        len_data_loader = iter_per_epoch
    else:
        len_data_loader, iter_per_epoch = len(data_loader), None

    for data_iter_step, (image1, image2, gt, pairname) in enumerate(metric_logger.log_every(data_loader, print_freq, header, max_iter=iter_per_epoch)):
        
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, data_iter_step / len_data_loader + epoch, args)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            prediction = model(image1, image2)
            prediction, conf = split_prediction_conf(prediction, criterion.with_conf)
            batch_metrics = metrics(prediction.detach(), gt)
            loss = criterion(prediction, gt) if conf is None else criterion(prediction, gt, conf)
            
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
        for k,v in batch_metrics.items():
            metric_logger.update(**{k: v.item()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        #if args.dsitributed: loss_value_reduce = misc.all_reduce_mean(loss_value)
        time_to_log = ((data_iter_step + 1) % (args.tboard_log_step * accum_iter) == 0 or data_iter_step == len_data_loader-1)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and time_to_log:
            epoch_1000x = int((data_iter_step / len_data_loader + epoch) * 1000)
            # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            for k,v in batch_metrics.items():
                log_writer.add_scalar('train/'+k, v.item(), epoch_1000x)

    # gather the stats from all processes
    #if args.distributed: metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_one_epoch(model: torch.nn.Module,
                   criterion: torch.nn.Module,
                   metrics: torch.nn.Module,
                   data_loaders: list[Iterable],
                   device: torch.device,
                   epoch: int,
                   log_writer=None,
                   args=None):

    model.eval()
    metric_loggers = []
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    conf_mode = args.tile_conf_mode
    crop = args.crop
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    results = {}
    dnames = []
    image1, image2, gt, prediction = None, None, None, None
    for didx, data_loader in enumerate(data_loaders):
        dname = str(data_loader.dataset)
        dnames.append(dname)
        metric_loggers.append(misc.MetricLogger(delimiter="  "))
        for data_iter_step, (image1, image2, gt, pairname) in enumerate(metric_loggers[didx].log_every(data_loader, print_freq, header)):
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            if dname.startswith('Spring'):
                assert gt.size(2)==image1.size(2)*2 and gt.size(3)==image1.size(3)*2
                gt = (gt[:,:,0::2,0::2] + gt[:,:,0::2,1::2] + gt[:,:,1::2,0::2] + gt[:,:,1::2,1::2] ) / 4.0 # we approximate the gt based on the 2x upsampled ones

            with torch.inference_mode():
                prediction, tiled_loss, c = tiled_pred(model, criterion, image1, image2, gt, conf_mode=conf_mode, overlap=args.val_overlap, crop=crop, with_conf=criterion.with_conf)
                batch_metrics = metrics(prediction.detach(), gt)
                loss = criterion(prediction.detach(), gt) if not criterion.with_conf else criterion(prediction.detach(), gt, c)
                loss_value = loss.item()
                metric_loggers[didx].update(loss_tiled=tiled_loss.item())
                metric_loggers[didx].update(**{f'loss': loss_value})
                for k,v in batch_metrics.items():
                    metric_loggers[didx].update(**{dname+'_' + k: v.item()})
        
    results = {k: meter.global_avg for ml in metric_loggers for k, meter in ml.meters.items()}
    if len(dnames)>1:
        for k in batch_metrics.keys():
            results['AVG_'+k] = sum(results[dname+'_'+k] for dname in dnames) / len(dnames)
            
    if log_writer is not None :
        epoch_1000x = int((1 + epoch) * 1000)
        for k,v in results.items():
            log_writer.add_scalar('val/'+k, v, epoch_1000x)

    print("Averaged stats:", results)
    return results

import torch.nn.functional as F
def _resize_img(img, new_size):
    return F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)
def _resize_stereo_or_flow(data, new_size):
    assert data.ndim==4
    assert data.size(1) in [1,2]
    scale_x = new_size[1]/float(data.size(3))
    out = F.interpolate(data, size=new_size, mode='bicubic', align_corners=False)
    out[:,0,:,:] *= scale_x
    if out.size(1)==2:
        scale_y = new_size[0]/float(data.size(2))        
        out[:,1,:,:] *= scale_y
        print(scale_x, new_size, data.shape)
    return out
    

@torch.no_grad()
def tiled_pred(model, criterion, img1, img2, gt,
               overlap=0.5, bad_crop_thr=0.05,
               downscale=False, crop=512, ret='loss',
               conf_mode='conf_expsigmoid_10_5', with_conf=False, 
               return_time=False):
                     
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged
    if gt is not None:
        B, C, H, W = gt.shape
    else:
        B, _, H, W = img1.shape
        C = model.head.num_channels-int(with_conf)
    win_height, win_width = crop[0], crop[1]
    
    # upscale to be larger than the crop
    do_change_scale =  H<win_height or W<win_width
    if do_change_scale: 
        upscale_factor = max(win_width/W, win_height/W)
        original_size = (H,W)
        new_size = (round(H*upscale_factor),round(W*upscale_factor))
        img1 = _resize_img(img1, new_size)
        img2 = _resize_img(img2, new_size)
        # resize gt just for the computation of tiled losses
        if gt is not None: gt = _resize_stereo_or_flow(gt, new_size)
        H,W = img1.shape[2:4]
        
    if conf_mode.startswith('conf_expsigmoid_'): # conf_expsigmoid_30_10
        beta, betasigmoid = map(float, conf_mode[len('conf_expsigmoid_'):].split('_'))
    elif conf_mode.startswith('conf_expbeta'): # conf_expbeta3
        beta = float(conf_mode[len('conf_expbeta'):])
    else:
        raise NotImplementedError(f"conf_mode {conf_mode} is not implemented")

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True

    # keep track of weighted sum of prediction*weights and weights
    accu_pred = img1.new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
    accu_conf = img1.new_zeros((B, H, W)) + 1e-16 # accumulate the weights 
    accu_c = img1.new_zeros((B, H, W)) # accumulate the weighted sum of confidences ; not so useful except for computing some losses

    tiled_losses = []
    
    if return_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        pred =  model(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))
        pred, predconf = split_prediction_conf(pred, with_conf=with_conf)
        
        if gt is not None: gtcrop = _crop(gt,sy1,sx1)
        if criterion is not None and gt is not None: 
            tiled_losses.append( criterion(pred, gtcrop).item() if predconf is None else criterion(pred, gtcrop, predconf).item() )
        
        if conf_mode.startswith('conf_expsigmoid_'):
            conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5)).view(B,win_height,win_width)
        elif conf_mode.startswith('conf_expbeta'):
            conf = torch.exp(- beta * predconf).view(B,win_height,win_width)
        else:
            raise NotImplementedError
                        
        accu_pred[...,sy1,sx1] += pred * conf[:,None,:,:]
        accu_conf[...,sy1,sx1] += conf
        accu_c[...,sy1,sx1] += predconf.view(B,win_height,win_width) * conf 
        
    pred = accu_pred / accu_conf[:, None,:,:]
    c = accu_c / accu_conf
    assert not torch.any(torch.isnan(pred))

    if return_time:
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)/1000.0 # this was in milliseconds

    if do_change_scale:
        pred = _resize_stereo_or_flow(pred, original_size)
    
    if return_time:
        return pred, torch.mean(torch.tensor(tiled_losses)), c, time
    return pred, torch.mean(torch.tensor(tiled_losses)), c


def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)

def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]