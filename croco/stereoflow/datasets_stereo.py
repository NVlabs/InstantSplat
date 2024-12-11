# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Dataset structure for stereo
# --------------------------------------------------------

import sys, os
import os.path as osp
import pickle
import numpy as np
from PIL import Image
import json
import h5py
from glob import glob
import cv2

import torch
from torch.utils import data

from .augmentor import StereoAugmentor



dataset_to_root = {
    'CREStereo': './data/stereoflow//crenet_stereo_trainset/stereo_trainset/crestereo/',
    'SceneFlow': './data/stereoflow//SceneFlow/',
    'ETH3DLowRes': './data/stereoflow/eth3d_lowres/',
    'Booster': './data/stereoflow/booster_gt/',
    'Middlebury2021': './data/stereoflow/middlebury/2021/data/',
    'Middlebury2014': './data/stereoflow/middlebury/2014/',
    'Middlebury2006': './data/stereoflow/middlebury/2006/',
    'Middlebury2005': './data/stereoflow/middlebury/2005/train/',
    'MiddleburyEval3':  './data/stereoflow/middlebury/MiddEval3/',
    'Spring': './data/stereoflow/spring/', 
    'Kitti15': './data/stereoflow/kitti-stereo-2015/',
    'Kitti12': './data/stereoflow/kitti-stereo-2012/',
}
cache_dir = "./data/stereoflow/datasets_stereo_cache/"


in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
def img_to_tensor(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
    img = (img-in1k_mean)/in1k_std
    return img
def disp_to_tensor(disp):
    return torch.from_numpy(disp)[None,:,:]

class StereoDataset(data.Dataset):
    
    def __init__(self, split, augmentor=False, crop_size=None, totensor=True):
        self.split = split
        if not augmentor: assert crop_size is None 
        if crop_size: assert augmentor
        self.crop_size = crop_size
        self.augmentor_str = augmentor
        self.augmentor = StereoAugmentor(crop_size) if augmentor else None
        self.totensor = totensor
        self.rmul = 1 # keep track of rmul
        self.has_constant_resolution = True # whether the dataset has constant resolution or not (=> don't use batch_size>1 at test time)
        self._prepare_data()
        self._load_or_build_cache()
        
    def prepare_data(self):
        """
        to be defined for each dataset 
        """
        raise NotImplementedError 
        
    def __len__(self):
        return len(self.pairnames)
        
    def __getitem__(self, index):
        pairname = self.pairnames[index]
        
        # get filenames 
        Limgname = self.pairname_to_Limgname(pairname)
        Rimgname = self.pairname_to_Rimgname(pairname)
        Ldispname = self.pairname_to_Ldispname(pairname) if self.pairname_to_Ldispname is not None else None
        
        # load images and disparities
        Limg = _read_img(Limgname)
        Rimg = _read_img(Rimgname)
        disp = self.load_disparity(Ldispname) if Ldispname is not None else None
        
        # sanity check
        if disp is not None: assert np.all(disp>0) or self.name=="Spring", (self.name, pairname, Ldispname)
        
        # apply augmentations
        if self.augmentor is not None:
            Limg, Rimg, disp = self.augmentor(Limg, Rimg, disp, self.name)
        
        if self.totensor:
            Limg = img_to_tensor(Limg)
            Rimg = img_to_tensor(Rimg)
            if disp is None:
                disp = torch.tensor([]) # to allow dataloader batching with default collate_gn
            else:
                disp = disp_to_tensor(disp)
        
        return Limg, Rimg, disp, str(pairname)
        
    def __rmul__(self, v):
        self.rmul *= v
        self.pairnames = v * self.pairnames
        return self
        
    def __str__(self):
        return f'{self.__class__.__name__}_{self.split}'
        
    def __repr__(self):
        s = f'{self.__class__.__name__}(split={self.split}, augmentor={self.augmentor_str}, crop_size={str(self.crop_size)}, totensor={self.totensor})'
        if self.rmul==1:
            s+=f'\n\tnum pairs: {len(self.pairnames)}'
        else:
            s+=f'\n\tnum pairs: {len(self.pairnames)} ({len(self.pairnames)//self.rmul}x{self.rmul})'
        return s

    def _set_root(self):
        self.root = dataset_to_root[self.name]
        assert os.path.isdir(self.root), f"could not find root directory for dataset {self.name}: {self.root}"       

    def _load_or_build_cache(self):
        cache_file = osp.join(cache_dir, self.name+'.pkl')
        if osp.isfile(cache_file):
            with open(cache_file, 'rb') as fid:
                self.pairnames = pickle.load(fid)[self.split]
        else:
            tosave = self._build_cache()
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as fid:
                pickle.dump(tosave, fid)
            self.pairnames = tosave[self.split]
        
class CREStereoDataset(StereoDataset):

    def _prepare_data(self):
        self.name = 'CREStereo'
        self._set_root()
        assert self.split in ['train']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname+'_left.jpg')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname+'_right.jpg')
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, pairname+'_left.disp.png')
        self.pairname_to_str = lambda pairname: pairname
        self.load_disparity = _read_crestereo_disp
        
    
    def _build_cache(self):
        allpairs = [s+'/'+f[:-len('_left.jpg')] for s in sorted(os.listdir(self.root)) for f in sorted(os.listdir(self.root+'/'+s)) if f.endswith('_left.jpg')]
        assert len(allpairs)==200000, "incorrect parsing of pairs in CreStereo"
        tosave = {'train': allpairs}
        return tosave
        
class SceneFlowDataset(StereoDataset):

    def _prepare_data(self):
        self.name = "SceneFlow"
        self._set_root()
        assert self.split in ['train_finalpass','train_cleanpass','train_allpass','test_finalpass','test_cleanpass','test_allpass','test1of100_cleanpass','test1of100_finalpass']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname).replace('/left/','/right/')
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, pairname).replace('/frames_finalpass/','/disparity/').replace('/frames_cleanpass/','/disparity/')[:-4]+'.pfm'
        self.pairname_to_str = lambda pairname: pairname[:-4]
        self.load_disparity = _read_sceneflow_disp
        
    def _build_cache(self):
        trainpairs = []
        # driving
        pairs = sorted(glob(self.root+'Driving/frames_finalpass/*/*/*/left/*.png'))
        pairs = list(map(lambda x: x[len(self.root):], pairs))
        assert len(pairs) == 4400, "incorrect parsing of pairs in SceneFlow"
        trainpairs += pairs
        # monkaa
        pairs = sorted(glob(self.root+'Monkaa/frames_finalpass/*/left/*.png'))
        pairs = list(map(lambda x: x[len(self.root):], pairs))
        assert len(pairs) == 8664, "incorrect parsing of pairs in SceneFlow"
        trainpairs += pairs
        # flyingthings
        pairs = sorted(glob(self.root+'FlyingThings/frames_finalpass/TRAIN/*/*/left/*.png'))
        pairs = list(map(lambda x: x[len(self.root):], pairs))
        assert len(pairs) == 22390, "incorrect parsing of pairs in SceneFlow"
        trainpairs += pairs
        assert len(trainpairs) == 35454, "incorrect parsing of pairs in SceneFlow"
        testpairs = sorted(glob(self.root+'FlyingThings/frames_finalpass/TEST/*/*/left/*.png'))
        testpairs = list(map(lambda x: x[len(self.root):], testpairs))
        assert len(testpairs) == 4370, "incorrect parsing of pairs in SceneFlow"
        test1of100pairs = testpairs[::100]
        assert len(test1of100pairs) == 44, "incorrect parsing of pairs in SceneFlow"
        # all 
        tosave = {'train_finalpass': trainpairs,
                  'train_cleanpass': list(map(lambda x: x.replace('frames_finalpass','frames_cleanpass'), trainpairs)),
                  'test_finalpass': testpairs,
                  'test_cleanpass': list(map(lambda x: x.replace('frames_finalpass','frames_cleanpass'), testpairs)),
                  'test1of100_finalpass': test1of100pairs,
                  'test1of100_cleanpass': list(map(lambda x: x.replace('frames_finalpass','frames_cleanpass'), test1of100pairs)),
                 }
        tosave['train_allpass'] = tosave['train_finalpass']+tosave['train_cleanpass']
        tosave['test_allpass'] = tosave['test_finalpass']+tosave['test_cleanpass']
        return tosave
   
class Md21Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Middlebury2021"
        self._set_root()
        assert self.split in ['train','subtrain','subval']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname.replace('/im0','/im1'))
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, pairname.split('/')[0], 'disp0.pfm')
        self.pairname_to_str = lambda pairname: pairname[:-4]
        self.load_disparity = _read_middlebury_disp
        
    def _build_cache(self):
        seqs = sorted(os.listdir(self.root))
        trainpairs = []
        for s in seqs:
            #trainpairs += [s+'/im0.png'] # we should remove it, it is included as such in other lightings
            trainpairs += [s+'/ambient/'+b+'/'+a for b in sorted(os.listdir(osp.join(self.root,s,'ambient'))) for a in sorted(os.listdir(osp.join(self.root,s,'ambient',b))) if a.startswith('im0')]
        assert len(trainpairs)==355
        subtrainpairs = [p for p in trainpairs if any(p.startswith(s+'/') for s in seqs[:-2])]
        subvalpairs = [p for p in trainpairs if any(p.startswith(s+'/') for s in seqs[-2:])]
        assert len(subtrainpairs)==335 and len(subvalpairs)==20, "incorrect parsing of pairs in Middlebury 2021"
        tosave = {'train': trainpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave 

class Md14Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Middlebury2014"
        self._set_root()
        assert self.split in ['train','subtrain','subval']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, osp.dirname(pairname), 'im0.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, osp.dirname(pairname), 'disp0.pfm')
        self.pairname_to_str = lambda pairname: pairname[:-4]
        self.load_disparity = _read_middlebury_disp
        self.has_constant_resolution = False
        
    def _build_cache(self):
        seqs = sorted(os.listdir(self.root))
        trainpairs = []
        for s in seqs:
            trainpairs += [s+'/im1.png',s+'/im1E.png',s+'/im1L.png']
        assert len(trainpairs)==138
        valseqs = ['Umbrella-imperfect','Vintage-perfect']
        assert all(s in seqs for s in valseqs)
        subtrainpairs = [p for p in trainpairs if not any(p.startswith(s+'/') for s in valseqs)]
        subvalpairs = [p for p in trainpairs if any(p.startswith(s+'/') for s in valseqs)]
        assert len(subtrainpairs)==132 and len(subvalpairs)==6, "incorrect parsing of pairs in Middlebury 2014"
        tosave = {'train': trainpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave 

class Md06Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Middlebury2006"
        self._set_root()
        assert self.split in ['train','subtrain','subval']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, osp.dirname(pairname), 'view5.png')
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, pairname.split('/')[0], 'disp1.png')
        self.load_disparity = _read_middlebury20052006_disp
        self.has_constant_resolution = False
        
    def _build_cache(self):
        seqs = sorted(os.listdir(self.root))
        trainpairs = []
        for s in seqs:
            for i in ['Illum1','Illum2','Illum3']:
                for e in ['Exp0','Exp1','Exp2']:
                    trainpairs.append(osp.join(s,i,e,'view1.png'))
        assert len(trainpairs)==189
        valseqs = ['Rocks1','Wood2']
        assert all(s in seqs for s in valseqs)
        subtrainpairs = [p for p in trainpairs if not any(p.startswith(s+'/') for s in valseqs)]
        subvalpairs = [p for p in trainpairs if any(p.startswith(s+'/') for s in valseqs)]
        assert len(subtrainpairs)==171 and len(subvalpairs)==18, "incorrect parsing of pairs in Middlebury 2006"
        tosave = {'train': trainpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave

class Md05Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Middlebury2005"
        self._set_root()
        assert self.split in ['train','subtrain','subval']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, osp.dirname(pairname), 'view5.png')
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, pairname.split('/')[0], 'disp1.png')
        self.pairname_to_str = lambda pairname: pairname[:-4]
        self.load_disparity = _read_middlebury20052006_disp
        
    def _build_cache(self):
        seqs = sorted(os.listdir(self.root))
        trainpairs = []
        for s in seqs:
            for i in ['Illum1','Illum2','Illum3']:
                for e in ['Exp0','Exp1','Exp2']:
                    trainpairs.append(osp.join(s,i,e,'view1.png'))
        assert len(trainpairs)==54, "incorrect parsing of pairs in Middlebury 2005"
        valseqs = ['Reindeer']
        assert all(s in seqs for s in valseqs)
        subtrainpairs = [p for p in trainpairs if not any(p.startswith(s+'/') for s in valseqs)]
        subvalpairs = [p for p in trainpairs if any(p.startswith(s+'/') for s in valseqs)]
        assert len(subtrainpairs)==45 and len(subvalpairs)==9, "incorrect parsing of pairs in Middlebury 2005"
        tosave = {'train': trainpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave
        
class MdEval3Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "MiddleburyEval3"
        self._set_root()
        assert self.split in [s+'_'+r for s in ['train','subtrain','subval','test','all'] for r in ['full','half','quarter']]
        if self.split.endswith('_full'):
            self.root = self.root.replace('/MiddEval3','/MiddEval3_F')
        elif self.split.endswith('_half'):        
            self.root = self.root.replace('/MiddEval3','/MiddEval3_H')
        else:
            assert self.split.endswith('_quarter')
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname, 'im0.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname, 'im1.png')
        self.pairname_to_Ldispname = lambda pairname: None if pairname.startswith('test') else osp.join(self.root, pairname, 'disp0GT.pfm')
        self.pairname_to_str = lambda pairname: pairname
        self.load_disparity = _read_middlebury_disp
        # for submission only
        self.submission_methodname = "CroCo-Stereo"
        self.submission_sresolution = 'F' if self.split.endswith('_full') else ('H' if self.split.endswith('_half') else 'Q')
        
    def _build_cache(self):
        trainpairs = ['train/'+s for s in sorted(os.listdir(self.root+'train/'))]
        testpairs = ['test/'+s for s in sorted(os.listdir(self.root+'test/'))]
        subvalpairs = trainpairs[-1:]
        subtrainpairs = trainpairs[:-1]
        allpairs = trainpairs+testpairs
        assert len(trainpairs)==15 and len(testpairs)==15 and len(subvalpairs)==1 and len(subtrainpairs)==14 and len(allpairs)==30, "incorrect parsing of pairs in Middlebury Eval v3"
        tosave = {}
        for r in ['full','half','quarter']:
            tosave.update(**{'train_'+r: trainpairs, 'subtrain_'+r: subtrainpairs, 'subval_'+r: subvalpairs, 'test_'+r: testpairs, 'all_'+r: allpairs})
        return tosave
        
    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==2
        assert prediction.dtype==np.float32
        outfile = os.path.join(outdir, pairname.split('/')[0].replace('train','training')+self.submission_sresolution, pairname.split('/')[1], 'disp0'+self.submission_methodname+'.pfm')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writePFM(outfile, prediction)
        timefile = os.path.join( os.path.dirname(outfile), "time"+self.submission_methodname+'.txt')
        with open(timefile, 'w') as fid:
            fid.write(str(time))

    def finalize_submission(self, outdir):
        cmd = f'cd {outdir}/; zip -r "{self.submission_methodname}.zip" .'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/{self.submission_methodname}.zip')

class ETH3DLowResDataset(StereoDataset):

    def _prepare_data(self):
        self.name = "ETH3DLowRes"
        self._set_root()
        assert self.split in ['train','test','subtrain','subval','all']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname, 'im0.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname, 'im1.png')
        self.pairname_to_Ldispname = None if self.split=='test' else lambda pairname: None if pairname.startswith('test/') else osp.join(self.root, pairname.replace('train/','train_gt/'), 'disp0GT.pfm')
        self.pairname_to_str = lambda pairname: pairname
        self.load_disparity = _read_eth3d_disp
        self.has_constant_resolution = False
        
    def _build_cache(self):
        trainpairs = ['train/' + s for s in sorted(os.listdir(self.root+'train/'))]
        testpairs = ['test/' + s for s in sorted(os.listdir(self.root+'test/'))]
        assert len(trainpairs) == 27 and len(testpairs) == 20, "incorrect parsing of pairs in ETH3D Low Res"
        subvalpairs = ['train/delivery_area_3s','train/electro_3l','train/playground_3l']
        assert all(p in trainpairs for p in subvalpairs)
        subtrainpairs = [p for p in trainpairs if not p in subvalpairs]
        assert len(subvalpairs)==3 and len(subtrainpairs)==24, "incorrect parsing of pairs in ETH3D Low Res"
        tosave = {'train': trainpairs, 'test': testpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs, 'all': trainpairs+testpairs}
        return tosave

    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==2
        assert prediction.dtype==np.float32
        outfile = os.path.join(outdir, 'low_res_two_view', pairname.split('/')[1]+'.pfm')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writePFM(outfile, prediction)
        timefile = outfile[:-4]+'.txt'
        with open(timefile, 'w') as fid:
            fid.write('runtime '+str(time))

    def finalize_submission(self, outdir):
        cmd = f'cd {outdir}/; zip -r "eth3d_low_res_two_view_results.zip" low_res_two_view'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/eth3d_low_res_two_view_results.zip')

class BoosterDataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Booster"
        self._set_root()
        assert self.split in ['train_balanced','test_balanced','subtrain_balanced','subval_balanced'] # we use only the balanced version
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname)
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname).replace('/camera_00/','/camera_02/')
        self.pairname_to_Ldispname = lambda pairname: osp.join(self.root, osp.dirname(pairname), '../disp_00.npy') # same images with different colors, same gt per sequence
        self.pairname_to_str = lambda pairname: pairname[:-4].replace('/camera_00/','/')
        self.load_disparity = _read_booster_disp
        
        
    def _build_cache(self):
        trainseqs = sorted(os.listdir(self.root+'train/balanced'))
        trainpairs = ['train/balanced/'+s+'/camera_00/'+imname for s in trainseqs for imname in sorted(os.listdir(self.root+'train/balanced/'+s+'/camera_00/'))]
        testpairs = ['test/balanced/'+s+'/camera_00/'+imname for s in sorted(os.listdir(self.root+'test/balanced')) for imname in sorted(os.listdir(self.root+'test/balanced/'+s+'/camera_00/'))]
        assert len(trainpairs) == 228 and len(testpairs) == 191
        subtrainpairs = [p for p in trainpairs if any(s in p for s in trainseqs[:-2])]
        subvalpairs = [p for p in trainpairs if any(s in p for s in trainseqs[-2:])]
        # warning: if we do validation split, we should split scenes!!!
        tosave = {'train_balanced': trainpairs, 'test_balanced': testpairs, 'subtrain_balanced': subtrainpairs, 'subval_balanced': subvalpairs,}
        return tosave
        
class SpringDataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Spring"
        self._set_root()
        assert self.split in ['train', 'test', 'subtrain', 'subval']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname+'.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname+'.png').replace('frame_right','<frame_right>').replace('frame_left','frame_right').replace('<frame_right>','frame_left')
        self.pairname_to_Ldispname = lambda pairname: None if pairname.startswith('test') else osp.join(self.root, pairname+'.dsp5').replace('frame_left','disp1_left').replace('frame_right','disp1_right')
        self.pairname_to_str = lambda pairname: pairname
        self.load_disparity = _read_hdf5_disp        
        
    def _build_cache(self):
        trainseqs = sorted(os.listdir( osp.join(self.root,'train')))
        trainpairs = [osp.join('train',s,'frame_left',f[:-4]) for s in trainseqs for f in sorted(os.listdir(osp.join(self.root,'train',s,'frame_left')))]
        testseqs = sorted(os.listdir( osp.join(self.root,'test')))
        testpairs = [osp.join('test',s,'frame_left',f[:-4]) for s in testseqs for f in sorted(os.listdir(osp.join(self.root,'test',s,'frame_left')))]
        testpairs += [p.replace('frame_left','frame_right') for p in testpairs]
        """maxnorm = {'0001': 32.88, '0002': 228.5, '0004': 298.2, '0005': 142.5, '0006': 113.6, '0007': 27.3, '0008': 554.5, '0009': 155.6, '0010': 126.1, '0011': 87.6, '0012': 303.2, '0013': 24.14, '0014': 82.56, '0015': 98.44, '0016': 156.9, '0017': 28.17, '0018': 21.03, '0020': 178.0, '0021': 58.06, '0022': 354.2, '0023': 8.79, '0024': 97.06, '0025': 55.16, '0026': 91.9, '0027': 156.6, '0030': 200.4, '0032': 58.66, '0033': 373.5, '0036': 149.4, '0037': 5.625, '0038': 37.0, '0039': 12.2, '0041': 453.5, '0043': 457.0, '0044': 379.5, '0045': 161.8, '0047': 105.44} # => let'use 0041"""
        subtrainpairs = [p for p in trainpairs if p.split('/')[1]!='0041']
        subvalpairs = [p for p in trainpairs if p.split('/')[1]=='0041']
        assert len(trainpairs)==5000 and len(testpairs)==2000 and len(subtrainpairs)==4904 and len(subvalpairs)==96, "incorrect parsing of pairs in Spring"
        tosave = {'train': trainpairs, 'test': testpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave
        
    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==2
        assert prediction.dtype==np.float32
        outfile = os.path.join(outdir, pairname+'.dsp5').replace('frame_left','disp1_left').replace('frame_right','disp1_right')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writeDsp5File(prediction, outfile)
        
    def finalize_submission(self, outdir):
        assert self.split=='test'
        exe = "{self.root}/disp1_subsampling"
        if os.path.isfile(exe):
            cmd = f'cd "{outdir}/test"; {exe} .'
            print(cmd)
            os.system(cmd)
        else:
            print('Could not find disp1_subsampling executable for submission.')
            print('Please download it and run:')
            print(f'cd "{outdir}/test"; <disp1_subsampling_exe> .')

class Kitti12Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Kitti12"
        self._set_root()
        assert self.split in ['train','test']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname+'_10.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname.replace('/colored_0/','/colored_1/')+'_10.png')
        self.pairname_to_Ldispname = None if self.split=='test' else lambda pairname: osp.join(self.root, pairname.replace('/colored_0/','/disp_occ/')+'_10.png')
        self.pairname_to_str = lambda pairname: pairname.replace('/colored_0/','/')
        self.load_disparity = _read_kitti_disp
        
    def _build_cache(self):
        trainseqs = ["training/colored_0/%06d"%(i) for i in range(194)]
        testseqs = ["testing/colored_0/%06d"%(i) for i in range(195)]
        assert len(trainseqs)==194 and len(testseqs)==195, "incorrect parsing of pairs in Kitti12"
        tosave = {'train': trainseqs, 'test': testseqs}
        return tosave 

    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==2
        assert prediction.dtype==np.float32
        outfile = os.path.join(outdir, pairname.split('/')[-1]+'_10.png')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        img = (prediction * 256).astype('uint16')
        Image.fromarray(img).save(outfile)

    def finalize_submission(self, outdir):
        assert self.split=='test'
        cmd = f'cd {outdir}/; zip -r "kitti12_results.zip" .'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/kitti12_results.zip')

class Kitti15Dataset(StereoDataset):

    def _prepare_data(self):
        self.name = "Kitti15"
        self._set_root()
        assert self.split in ['train','subtrain','subval','test']
        self.pairname_to_Limgname = lambda pairname: osp.join(self.root, pairname+'_10.png')
        self.pairname_to_Rimgname = lambda pairname: osp.join(self.root, pairname.replace('/image_2/','/image_3/')+'_10.png')
        self.pairname_to_Ldispname = None if self.split=='test' else lambda pairname: osp.join(self.root, pairname.replace('/image_2/','/disp_occ_0/')+'_10.png')
        self.pairname_to_str = lambda pairname: pairname.replace('/image_2/','/')
        self.load_disparity = _read_kitti_disp
        
    def _build_cache(self):
        trainseqs = ["training/image_2/%06d"%(i) for i in range(200)]
        subtrainseqs = trainseqs[:-5]
        subvalseqs = trainseqs[-5:]
        testseqs = ["testing/image_2/%06d"%(i) for i in range(200)]
        assert len(trainseqs)==200 and len(subtrainseqs)==195 and len(subvalseqs)==5 and len(testseqs)==200, "incorrect parsing of pairs in Kitti15"
        tosave = {'train': trainseqs, 'subtrain': subtrainseqs, 'subval': subvalseqs, 'test': testseqs}
        return tosave 

    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==2
        assert prediction.dtype==np.float32
        outfile = os.path.join(outdir, 'disp_0', pairname.split('/')[-1]+'_10.png')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        img = (prediction * 256).astype('uint16')
        Image.fromarray(img).save(outfile)

    def finalize_submission(self, outdir):
        assert self.split=='test'
        cmd = f'cd {outdir}/; zip -r "kitti15_results.zip" disp_0'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/kitti15_results.zip')


### auxiliary functions

def _read_img(filename):
    # convert to RGB for scene flow finalpass data
    img = np.asarray(Image.open(filename).convert('RGB'))
    return img

def _read_booster_disp(filename):
    disp = np.load(filename)
    disp[disp==0.0] = np.inf
    return disp

def _read_png_disp(filename, coef=1.0):
    disp = np.asarray(Image.open(filename))
    disp = disp.astype(np.float32) / coef
    disp[disp==0.0] = np.inf
    return disp 

def _read_pfm_disp(filename):
    disp = np.ascontiguousarray(_read_pfm(filename)[0])
    disp[disp<=0] = np.inf # eg /nfs/data/ffs-3d/datasets/middlebury/2014/Shopvac-imperfect/disp0.pfm
    return disp

def _read_npy_disp(filename):
    return np.load(filename)

def _read_crestereo_disp(filename): return _read_png_disp(filename, coef=32.0)
def _read_middlebury20052006_disp(filename): return _read_png_disp(filename, coef=1.0)
def _read_kitti_disp(filename): return _read_png_disp(filename, coef=256.0)
_read_sceneflow_disp = _read_pfm_disp
_read_eth3d_disp = _read_pfm_disp
_read_middlebury_disp = _read_pfm_disp
_read_carla_disp = _read_pfm_disp
_read_tartanair_disp = _read_npy_disp
    
def _read_hdf5_disp(filename):
    disp = np.asarray(h5py.File(filename)['disparity'])
    disp[np.isnan(disp)] = np.inf # make invalid values as +inf
    #disp[disp==0.0] = np.inf # make invalid values as +inf
    return disp.astype(np.float32)
    
import re
def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)

def writeDsp5File(disp, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)


# disp visualization

def vis_disparity(disp, m=None, M=None):
    if m is None: m = disp.min()
    if M is None: M = disp.max()
    disp_vis = (disp - m) / (M-m) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    return disp_vis

# dataset getter 
    
def get_train_dataset_stereo(dataset_str, augmentor=True, crop_size=None):
    dataset_str = dataset_str.replace('(','Dataset(')
    if augmentor:
        dataset_str = dataset_str.replace(')',', augmentor=True)')
    if crop_size is not None:
        dataset_str = dataset_str.replace(')',', crop_size={:s})'.format(str(crop_size)))
    return eval(dataset_str)
    
def get_test_datasets_stereo(dataset_str):
    dataset_str = dataset_str.replace('(','Dataset(')
    return [eval(s) for s in dataset_str.split('+')]