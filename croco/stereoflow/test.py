# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Main test function
# --------------------------------------------------------

import os
import argparse
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import utils.misc as misc
from models.croco_downstream import CroCoDownstreamBinocular
from models.head_downstream import PixelwiseTaskWithDPT

from stereoflow.criterion import *
from stereoflow.datasets_stereo import get_test_datasets_stereo
from stereoflow.datasets_flow import get_test_datasets_flow
from stereoflow.engine import tiled_pred

from stereoflow.datasets_stereo import vis_disparity
from stereoflow.datasets_flow import flowToColor

def get_args_parser():
    parser = argparse.ArgumentParser('Test CroCo models on stereo/flow', add_help=False)
    # important argument 
    parser.add_argument('--model', required=True, type=str, help='Path to the model to evaluate')
    parser.add_argument('--dataset', required=True, type=str, help="test dataset (there can be multiple dataset separated by a +)")
    # tiling 
    parser.add_argument('--tile_conf_mode', type=str, default='', help='Weights for the tiling aggregation based on confidence (empty means use the formula from the loaded checkpoint')
    parser.add_argument('--tile_overlap', type=float, default=0.7, help='overlap between tiles')
    # save (it will automatically go to <model_path>_<dataset_str>/<tile_str>_<save>)
    parser.add_argument('--save', type=str, nargs='+', default=[], 
                        help='what to save: \
                              metrics (pickle file), \
                              pred (raw prediction save as torch tensor), \
                              visu (visualization in png of each prediction), \
                              err10 (visualization in png of the error clamp at 10 for each prediction), \
                              submission (submission file)')
    # other (no impact)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser
    
    
def _load_model_and_criterion(model_path, do_load_metrics, device):
    print('loading model from', model_path)
    assert os.path.isfile(model_path)
    ckpt = torch.load(model_path, 'cpu')
    
    ckpt_args = ckpt['args']
    task = ckpt_args.task
    tile_conf_mode = ckpt_args.tile_conf_mode
    num_channels = {'stereo': 1, 'flow': 2}[task]
    with_conf =  eval(ckpt_args.criterion).with_conf
    if with_conf: num_channels += 1
    print('head: PixelwiseTaskWithDPT()')
    head = PixelwiseTaskWithDPT()
    head.num_channels = num_channels
    print('croco_args:', ckpt_args.croco_args)
    model = CroCoDownstreamBinocular(head, **ckpt_args.croco_args)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    model = model.to(device)
    
    if do_load_metrics:
        if task=='stereo':
            metrics = StereoDatasetMetrics().to(device)
        else:
            metrics = FlowDatasetMetrics().to(device)
    else:
        metrics = None
    
    return model, metrics, ckpt_args.crop, with_conf, task, tile_conf_mode
    
    
def _save_batch(pred, gt, pairnames, dataset, task, save, outdir, time, submission_dir=None):

    for i in range(len(pairnames)):
        
        pairname = eval(pairnames[i]) if pairnames[i].startswith('(') else pairnames[i] # unbatch pairname 
        fname = os.path.join(outdir, dataset.pairname_to_str(pairname))
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        predi = pred[i,...]
        if gt is not None: gti = gt[i,...]
        
        if 'pred' in save:
            torch.save(predi.squeeze(0).cpu(), fname+'_pred.pth')
            
        if 'visu' in save:
            if task=='stereo':
                disparity = predi.permute((1,2,0)).squeeze(2).cpu().numpy()
                m,M = None
                if gt is not None:
                    mask = torch.isfinite(gti)
                    m = gt[mask].min()
                    M = gt[mask].max()
                img_disparity = vis_disparity(disparity, m=m, M=M)
                Image.fromarray(img_disparity).save(fname+'_pred.png')
            else:
                # normalize flowToColor according to the maxnorm of gt (or prediction if not available)
                flowNorm = torch.sqrt(torch.sum( (gti if gt is not None else predi)**2, dim=0)).max().item()
                imgflow = flowToColor(predi.permute((1,2,0)).cpu().numpy(), maxflow=flowNorm)
                Image.fromarray(imgflow).save(fname+'_pred.png')
                
        if 'err10' in save:
            assert gt is not None
            L2err = torch.sqrt(torch.sum( (gti-predi)**2, dim=0))
            valid = torch.isfinite(gti[0,:,:])
            L2err[~valid] = 0.0
            L2err = torch.clamp(L2err, max=10.0)
            red = (L2err*255.0/10.0).to(dtype=torch.uint8)[:,:,None]
            zer = torch.zeros_like(red)
            imgerr = torch.cat( (red,zer,zer), dim=2).cpu().numpy()
            Image.fromarray(imgerr).save(fname+'_err10.png')
            
        if 'submission' in save:
            assert submission_dir is not None
            predi_np = predi.permute(1,2,0).squeeze(2).cpu().numpy() # transform into HxWx2 for flow or HxW for stereo
            dataset.submission_save_pairname(pairname, predi_np, submission_dir, time)

def main(args):
        
    # load the pretrained model and metrics
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model, metrics, cropsize, with_conf, task, tile_conf_mode = _load_model_and_criterion(args.model, 'metrics' in args.save, device)
    if args.tile_conf_mode=='': args.tile_conf_mode = tile_conf_mode
    
    # load the datasets 
    datasets = (get_test_datasets_stereo if task=='stereo' else get_test_datasets_flow)(args.dataset)
    dataloaders = [DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False) for dataset in datasets]    
       
    # run
    for i,dataloader in enumerate(dataloaders):
        dataset = datasets[i]
        dstr = args.dataset.split('+')[i]
        
        outdir = args.model+'_'+misc.filename(dstr)
        if 'metrics' in args.save and len(args.save)==1:
            fname = os.path.join(outdir, f'conf_{args.tile_conf_mode}_overlap_{args.tile_overlap}.pkl')
            if os.path.isfile(fname) and len(args.save)==1:
                print('  metrics already compute in '+fname)
                with open(fname, 'rb') as fid:
                    results = pickle.load(fid)
                for k,v in results.items():
                    print('{:s}: {:.3f}'.format(k, v))
                continue
                        
        if 'submission' in args.save:
            dirname = f'submission_conf_{args.tile_conf_mode}_overlap_{args.tile_overlap}'
            submission_dir = os.path.join(outdir, dirname)
        else:
            submission_dir = None
           
        print('')
        print('saving {:s} in {:s}'.format('+'.join(args.save), outdir))
        print(repr(dataset))
    
        if metrics is not None: 
            metrics.reset()
                
        for data_iter_step, (image1, image2, gt, pairnames) in enumerate(tqdm(dataloader)):
        
            do_flip = (task=='stereo' and dstr.startswith('Spring') and any("right" in p for p in pairnames)) # we flip the images and will flip the prediction after as we assume img1 is on the left 
            
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True) if gt.numel()>0 else None # special case for test time
            if do_flip:
                assert all("right" in p for p in pairnames) 
                image1 = image1.flip(dims=[3]) # this is already the right frame, let's flip it
                image2 = image2.flip(dims=[3])
                gt = gt # that is ok
                        
            with torch.inference_mode():
                pred, _, _, time = tiled_pred(model, None, image1, image2, None if dataset.name=='Spring' else gt, conf_mode=args.tile_conf_mode, overlap=args.tile_overlap, crop=cropsize, with_conf=with_conf, return_time=True)

                if do_flip:
                    pred = pred.flip(dims=[3])
                
                if metrics is not None: 
                    metrics.add_batch(pred, gt)
                
                if any(k in args.save for k in ['pred','visu','err10','submission']):
                    _save_batch(pred, gt, pairnames, dataset, task, args.save, outdir, time, submission_dir=submission_dir)                
            

        # print 
        if metrics is not None: 
            results = metrics.get_results()
            for k,v in results.items():
                print('{:s}: {:.3f}'.format(k, v))
                
        # save if needed
        if 'metrics' in args.save:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fid:
                pickle.dump(results, fid)
            print('metrics saved in', fname)
            
        # finalize submission if needed
        if 'submission' in args.save:
            dataset.finalize_submission(submission_dir)
                
        
            
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)