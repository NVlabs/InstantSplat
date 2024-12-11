# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Losses, metrics per batch, metrics per dataset 
# --------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

def _get_gtnorm(gt):
    if gt.size(1)==1: # stereo
        return gt
    # flow 
    return torch.sqrt(torch.sum(gt**2, dim=1, keepdims=True)) # Bx1xHxW

############ losses without confidence

class L1Loss(nn.Module):
    
    def __init__(self, max_gtnorm=None):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = False 
    
    def _error(self, gt, predictions):
        return torch.abs(gt-predictions)
    
    def forward(self, predictions, gt, inspect=False):
        mask = torch.isfinite(gt)
        if self.max_gtnorm is not None: 
            mask *= _get_gtnorm(gt).expand(-1,gt.size(1),-1,-1)<self.max_gtnorm
        if inspect:
            return self._error(gt, predictions)
        return self._error(gt[mask],predictions[mask]).mean()

############## losses with confience
## there are several parametrizations

class LaplacianLoss(nn.Module): # used for CroCo-Stereo on ETH3D, d'=exp(d)
    
    def __init__(self, max_gtnorm=None):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = True
        
    def forward(self, predictions, gt, conf):
        mask = torch.isfinite(gt)
        mask = mask[:,0,:,:]
        if self.max_gtnorm is not None: mask *= _get_gtnorm(gt)[:,0,:,:]<self.max_gtnorm
        conf = conf.squeeze(1)
        return ( torch.abs(gt-predictions).sum(dim=1)[mask] / torch.exp(conf[mask]) + conf[mask] ).mean()# + torch.log(2) => which is a constant


class LaplacianLossBounded(nn.Module): # used for CroCo-Flow ; in the equation of the paper, we have a=1/b
    def __init__(self, max_gtnorm=10000., a=0.25, b=4.):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = True
        self.a, self.b = a, b
        
    def forward(self, predictions, gt, conf):
        mask = torch.isfinite(gt)
        mask = mask[:,0,:,:]
        if self.max_gtnorm is not None: mask *= _get_gtnorm(gt)[:,0,:,:]<self.max_gtnorm
        conf = conf.squeeze(1)
        conf = (self.b - self.a) * torch.sigmoid(conf) + self.a
        return ( torch.abs(gt-predictions).sum(dim=1)[mask] / conf[mask] + torch.log(conf)[mask] ).mean()# + torch.log(2) => which is a constant

class LaplacianLossBounded2(nn.Module): # used for CroCo-Stereo (except for ETH3D) ; in the equation of the paper, we have a=b
    def __init__(self, max_gtnorm=None, a=3.0, b=3.0):
        super().__init__()
        self.max_gtnorm = max_gtnorm
        self.with_conf = True
        self.a, self.b = a, b
        
    def forward(self, predictions, gt, conf):
        mask = torch.isfinite(gt)
        mask = mask[:,0,:,:]
        if self.max_gtnorm is not None: mask *= _get_gtnorm(gt)[:,0,:,:]<self.max_gtnorm
        conf = conf.squeeze(1)
        conf = 2 * self.a * (torch.sigmoid(conf / self.b) - 0.5 )
        return ( torch.abs(gt-predictions).sum(dim=1)[mask] / torch.exp(conf[mask]) + conf[mask] ).mean()# + torch.log(2) => which is a constant
        
############## metrics per batch 

class StereoMetrics(nn.Module):

    def __init__(self, do_quantile=False):
        super().__init__()
        self.bad_ths = [0.5,1,2,3]
        self.do_quantile = do_quantile
        
    def forward(self, predictions, gt):
        B = predictions.size(0)
        metrics = {}
        gtcopy = gt.clone() 
        mask = torch.isfinite(gtcopy)
        gtcopy[~mask] = 999999.0 # we make a copy and put a non-infinite value, such that it does not become nan once multiplied by the mask value 0
        Npx = mask.view(B,-1).sum(dim=1)
        L1error = (torch.abs(gtcopy-predictions)*mask).view(B,-1)
        L2error = (torch.square(gtcopy-predictions)*mask).view(B,-1)
        # avgerr
        metrics['avgerr'] = torch.mean(L1error.sum(dim=1)/Npx )
        # rmse
        metrics['rmse'] = torch.sqrt(L2error.sum(dim=1)/Npx).mean(dim=0)
        # err > t for t in [0.5,1,2,3]
        for ths in self.bad_ths:
            metrics['bad@{:.1f}'.format(ths)] = (((L1error>ths)* mask.view(B,-1)).sum(dim=1)/Npx).mean(dim=0) * 100
        return metrics
        
class FlowMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.bad_ths = [1,3,5]
        
    def forward(self, predictions, gt):
        B = predictions.size(0)        
        metrics = {}
        mask = torch.isfinite(gt[:,0,:,:]) # both x and y would be infinite
        Npx = mask.view(B,-1).sum(dim=1)
        gtcopy = gt.clone() # to compute L1/L2 error, we need to have non-infinite value, the error computed at this locations will be ignored
        gtcopy[:,0,:,:][~mask] = 999999.0
        gtcopy[:,1,:,:][~mask] = 999999.0
        L1error = (torch.abs(gtcopy-predictions).sum(dim=1)*mask).view(B,-1)
        L2error = (torch.sqrt(torch.sum(torch.square(gtcopy-predictions),dim=1))*mask).view(B,-1)
        metrics['L1err'] = torch.mean(L1error.sum(dim=1)/Npx )
        metrics['EPE'] = torch.mean(L2error.sum(dim=1)/Npx )
        for ths in self.bad_ths:
            metrics['bad@{:.1f}'.format(ths)] = (((L2error>ths)* mask.view(B,-1)).sum(dim=1)/Npx).mean(dim=0) * 100
        return metrics
        
############## metrics per dataset
## we update the average and maintain the number of pixels while adding data batch per batch 
## at the beggining, call reset()
## after each batch, call add_batch(...)
## at the end: call get_results()

class StereoDatasetMetrics(nn.Module):

    def __init__(self):
        super().__init__()
        self.bad_ths = [0.5,1,2,3]
        
    def reset(self):
        self.agg_N = 0 # number of pixels so far 
        self.agg_L1err = torch.tensor(0.0) # L1 error so far 
        self.agg_Nbad = [0 for _ in self.bad_ths] # counter of bad pixels 
        self._metrics = None
                
    def add_batch(self, predictions, gt):
        assert predictions.size(1)==1, predictions.size()
        assert gt.size(1)==1, gt.size()
        if gt.size(2)==predictions.size(2)*2 and gt.size(3)==predictions.size(3)*2: # special case for Spring ...
            L1err = torch.minimum( torch.minimum( torch.minimum(
                torch.sum(torch.abs(gt[:,:,0::2,0::2]-predictions),dim=1),
                torch.sum(torch.abs(gt[:,:,1::2,0::2]-predictions),dim=1)),
                torch.sum(torch.abs(gt[:,:,0::2,1::2]-predictions),dim=1)),
                torch.sum(torch.abs(gt[:,:,1::2,1::2]-predictions),dim=1))
            valid = torch.isfinite(L1err)
        else:
            valid = torch.isfinite(gt[:,0,:,:]) # both x and y would be infinite
            L1err = torch.sum(torch.abs(gt-predictions),dim=1)
        N = valid.sum()
        Nnew = self.agg_N + N
        self.agg_L1err = float(self.agg_N)/Nnew * self.agg_L1err + L1err[valid].mean().cpu() * float(N)/Nnew
        self.agg_N = Nnew
        for i,th in enumerate(self.bad_ths):
            self.agg_Nbad[i] += (L1err[valid]>th).sum().cpu()
   
    def _compute_metrics(self):
        if self._metrics is not None: return
        out = {}
        out['L1err'] = self.agg_L1err.item()
        for i,th in enumerate(self.bad_ths):
            out['bad@{:.1f}'.format(th)] = (float(self.agg_Nbad[i]) / self.agg_N).item() * 100.0
        self._metrics = out
        
    def get_results(self): 
        self._compute_metrics() # to avoid recompute them multiple times
        return self._metrics

class FlowDatasetMetrics(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bad_ths = [0.5,1,3,5]
        self.speed_ths = [(0,10),(10,40),(40,torch.inf)]
    
    def reset(self):
        self.agg_N = 0 # number of pixels so far 
        self.agg_L1err = torch.tensor(0.0) # L1 error so far 
        self.agg_L2err = torch.tensor(0.0) # L2 (=EPE) error so far 
        self.agg_Nbad = [0 for _ in self.bad_ths] # counter of bad pixels 
        self.agg_EPEspeed = [torch.tensor(0.0) for _ in self.speed_ths] # EPE per speed bin so far 
        self.agg_Nspeed = [0 for _ in self.speed_ths] # N pixels per speed bin so far
        self._metrics = None
        self.pairname_results = {}

    def add_batch(self, predictions, gt):
        assert predictions.size(1)==2, predictions.size()
        assert gt.size(1)==2, gt.size()
        if gt.size(2)==predictions.size(2)*2 and gt.size(3)==predictions.size(3)*2: # special case for Spring ...
            L1err = torch.minimum( torch.minimum( torch.minimum(
                torch.sum(torch.abs(gt[:,:,0::2,0::2]-predictions),dim=1),
                torch.sum(torch.abs(gt[:,:,1::2,0::2]-predictions),dim=1)),
                torch.sum(torch.abs(gt[:,:,0::2,1::2]-predictions),dim=1)),
                torch.sum(torch.abs(gt[:,:,1::2,1::2]-predictions),dim=1))
            L2err = torch.minimum( torch.minimum( torch.minimum(
                torch.sqrt(torch.sum(torch.square(gt[:,:,0::2,0::2]-predictions),dim=1)),
                torch.sqrt(torch.sum(torch.square(gt[:,:,1::2,0::2]-predictions),dim=1))),
                torch.sqrt(torch.sum(torch.square(gt[:,:,0::2,1::2]-predictions),dim=1))),
                torch.sqrt(torch.sum(torch.square(gt[:,:,1::2,1::2]-predictions),dim=1)))
            valid = torch.isfinite(L1err)
            gtspeed = (torch.sqrt(torch.sum(torch.square(gt[:,:,0::2,0::2]),dim=1)) + torch.sqrt(torch.sum(torch.square(gt[:,:,0::2,1::2]),dim=1)) +\
                       torch.sqrt(torch.sum(torch.square(gt[:,:,1::2,0::2]),dim=1)) + torch.sqrt(torch.sum(torch.square(gt[:,:,1::2,1::2]),dim=1)) ) / 4.0 # let's just average them
        else:
            valid = torch.isfinite(gt[:,0,:,:]) # both x and y would be infinite
            L1err = torch.sum(torch.abs(gt-predictions),dim=1)
            L2err = torch.sqrt(torch.sum(torch.square(gt-predictions),dim=1))
            gtspeed = torch.sqrt(torch.sum(torch.square(gt),dim=1))
        N = valid.sum()
        Nnew = self.agg_N + N
        self.agg_L1err = float(self.agg_N)/Nnew * self.agg_L1err + L1err[valid].mean().cpu() * float(N)/Nnew
        self.agg_L2err = float(self.agg_N)/Nnew * self.agg_L2err + L2err[valid].mean().cpu() * float(N)/Nnew
        self.agg_N = Nnew
        for i,th in enumerate(self.bad_ths):
            self.agg_Nbad[i] += (L2err[valid]>th).sum().cpu()
        for i,(th1,th2) in enumerate(self.speed_ths):
            vv = (gtspeed[valid]>=th1) * (gtspeed[valid]<th2)
            iNspeed = vv.sum()
            if iNspeed==0: continue
            iNnew = self.agg_Nspeed[i] + iNspeed
            self.agg_EPEspeed[i] = float(self.agg_Nspeed[i]) / iNnew * self.agg_EPEspeed[i] + float(iNspeed) / iNnew * L2err[valid][vv].mean().cpu()
            self.agg_Nspeed[i] = iNnew

    def _compute_metrics(self):
        if self._metrics is not None: return
        out = {}
        out['L1err'] = self.agg_L1err.item()
        out['EPE']  = self.agg_L2err.item()
        for i,th in enumerate(self.bad_ths):
            out['bad@{:.1f}'.format(th)] = (float(self.agg_Nbad[i]) / self.agg_N).item() * 100.0
        for i,(th1,th2) in enumerate(self.speed_ths):
            out['s{:d}{:s}'.format(th1, '-'+str(th2) if th2<torch.inf else '+')] = self.agg_EPEspeed[i].item()
        self._metrics = out
    
    def get_results(self): 
        self._compute_metrics() # to avoid recompute them multiple times
        return self._metrics