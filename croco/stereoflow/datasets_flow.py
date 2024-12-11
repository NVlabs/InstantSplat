# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Dataset structure for flow
# --------------------------------------------------------

import os
import os.path as osp
import pickle
import numpy as np
import struct
from PIL import Image 
import json
import h5py
import torch
from torch.utils import data

from .augmentor import FlowAugmentor
from .datasets_stereo import _read_img, img_to_tensor, dataset_to_root, _read_pfm
from copy import deepcopy
dataset_to_root = deepcopy(dataset_to_root)

dataset_to_root.update(**{
    'TartanAir': './data/stereoflow/TartanAir',
    'FlyingChairs': './data/stereoflow/FlyingChairs/',
    'FlyingThings': osp.join(dataset_to_root['SceneFlow'],'FlyingThings')+'/',
    'MPISintel': './data/stereoflow//MPI-Sintel/'+'/',
})
cache_dir = "./data/stereoflow/datasets_flow_cache/"


def flow_to_tensor(disp):
    return torch.from_numpy(disp).float().permute(2, 0, 1)

class FlowDataset(data.Dataset):
    
    def __init__(self, split, augmentor=False, crop_size=None, totensor=True):
        self.split = split
        if not augmentor: assert crop_size is None 
        if crop_size is not None: assert augmentor
        self.crop_size = crop_size
        self.augmentor_str = augmentor
        self.augmentor = FlowAugmentor(crop_size) if augmentor else None
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
        return len(self.pairnames) # each pairname is typically of the form (str, int1, int2) 
        
    def __getitem__(self, index):
        pairname = self.pairnames[index]
        
        # get filenames 
        img1name = self.pairname_to_img1name(pairname)
        img2name = self.pairname_to_img2name(pairname)
        flowname = self.pairname_to_flowname(pairname) if self.pairname_to_flowname is not None else None
        
        # load images and disparities
        img1 = _read_img(img1name)
        img2 = _read_img(img2name)
        flow = self.load_flow(flowname) if flowname is not None else None

        # apply augmentations
        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow, self.name)
        
        if self.totensor:
            img1 = img_to_tensor(img1)
            img2 = img_to_tensor(img2)
            if flow is not None: 
                flow = flow_to_tensor(flow)
            else:
                flow = torch.tensor([]) # to allow dataloader batching with default collate_gn
            pairname = str(pairname) # transform potential tuple to str to be able to batch it

        return img1, img2, flow, pairname
        
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

class TartanAirDataset(FlowDataset):

    def _prepare_data(self):
        self.name = "TartanAir"
        self._set_root()
        assert self.split in ['train']
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, pairname[0], 'image_left/{:06d}_left.png'.format(pairname[1]))
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, pairname[0], 'image_left/{:06d}_left.png'.format(pairname[2]))
        self.pairname_to_flowname = lambda pairname: osp.join(self.root, pairname[0], 'flow/{:06d}_{:06d}_flow.npy'.format(pairname[1],pairname[2]))
        self.pairname_to_str = lambda pairname: os.path.join(pairname[0][pairname[0].find('/')+1:], '{:06d}_{:06d}'.format(pairname[1], pairname[2]))
        self.load_flow = _read_numpy_flow
        
    def _build_cache(self):
        seqs = sorted(os.listdir(self.root))
        pairs = [(osp.join(s,s,difficulty,Pxxx),int(a[:6]),int(a[:6])+1) for s in seqs for difficulty in ['Easy','Hard'] for Pxxx in sorted(os.listdir(osp.join(self.root,s,s,difficulty))) for a in sorted(os.listdir(osp.join(self.root,s,s,difficulty,Pxxx,'image_left/')))[:-1]]
        assert len(pairs)==306268, "incorrect parsing of pairs in TartanAir"
        tosave = {'train': pairs}
        return tosave
        
class FlyingChairsDataset(FlowDataset):

    def _prepare_data(self):
        self.name = "FlyingChairs"
        self._set_root()
        assert self.split in ['train','val']
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, 'data', pairname+'_img1.ppm')
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, 'data', pairname+'_img2.ppm')
        self.pairname_to_flowname = lambda pairname: osp.join(self.root, 'data', pairname+'_flow.flo')
        self.pairname_to_str = lambda pairname: pairname
        self.load_flow = _read_flo_file
        
    def _build_cache(self):
        split_file = osp.join(self.root, 'chairs_split.txt')
        split_list = np.loadtxt(split_file, dtype=np.int32)
        trainpairs = ['{:05d}'.format(i) for i in np.where(split_list==1)[0]+1]
        valpairs = ['{:05d}'.format(i) for i in np.where(split_list==2)[0]+1]
        assert len(trainpairs)==22232 and len(valpairs)==640, "incorrect parsing of pairs in MPI-Sintel"
        tosave = {'train': trainpairs, 'val': valpairs}
        return tosave
        
class FlyingThingsDataset(FlowDataset):
    
    def _prepare_data(self):
        self.name = "FlyingThings"
        self._set_root()
        assert self.split in [f'{set_}_{pass_}pass{camstr}' for set_ in ['train','test','test1024'] for camstr in ['','_rightcam'] for pass_ in ['clean','final','all']]
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, f'frames_{pairname[3]}pass', pairname[0].replace('into_future','').replace('into_past',''), '{:04d}.png'.format(pairname[1]))
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, f'frames_{pairname[3]}pass', pairname[0].replace('into_future','').replace('into_past',''), '{:04d}.png'.format(pairname[2]))
        self.pairname_to_flowname = lambda pairname: osp.join(self.root, 'optical_flow', pairname[0], 'OpticalFlowInto{f:s}_{i:04d}_{c:s}.pfm'.format(f='Future' if 'future' in pairname[0] else 'Past', i=pairname[1], c='L' if 'left' in pairname[0] else 'R' ))
        self.pairname_to_str = lambda pairname: os.path.join(pairname[3]+'pass', pairname[0], 'Into{f:s}_{i:04d}_{c:s}'.format(f='Future' if 'future' in pairname[0] else 'Past',  i=pairname[1], c='L' if 'left' in pairname[0] else 'R' ))
        self.load_flow = _read_pfm_flow
        
    def _build_cache(self):
        tosave = {}
        # train and test splits for the different passes 
        for set_ in ['train', 'test']:
            sroot = osp.join(self.root, 'optical_flow', set_.upper())
            fname_to_i = lambda f: int(f[len('OpticalFlowIntoFuture_'):-len('_L.pfm')])
            pp = [(osp.join(set_.upper(), d, s, 'into_future/left'),fname_to_i(fname)) for d in sorted(os.listdir(sroot)) for s in sorted(os.listdir(osp.join(sroot,d))) for fname in sorted(os.listdir(osp.join(sroot,d, s, 'into_future/left')))[:-1]]
            pairs  = [(a,i,i+1) for a,i in pp]
            pairs += [(a.replace('into_future','into_past'),i+1,i) for a,i in pp]
            assert len(pairs)=={'train': 40302, 'test': 7866}[set_], "incorrect parsing of pairs Flying Things"
            for cam in ['left','right']:
                camstr = '' if cam=='left' else f'_{cam}cam'
                for pass_ in ['final', 'clean']:
                    tosave[f'{set_}_{pass_}pass{camstr}'] = [(a.replace('left',cam),i,j,pass_) for a,i,j in pairs]
                tosave[f'{set_}_allpass{camstr}'] = tosave[f'{set_}_cleanpass{camstr}'] + tosave[f'{set_}_finalpass{camstr}']
        # test1024: this is the same split as unimatch 'validation' split
        # see https://github.com/autonomousvision/unimatch/blob/master/dataloader/flow/datasets.py#L229
        test1024_nsamples = 1024
        alltest_nsamples = len(tosave['test_cleanpass'])  # 7866
        stride = alltest_nsamples // test1024_nsamples
        remove = alltest_nsamples % test1024_nsamples
        for cam in ['left','right']:
            camstr = '' if cam=='left' else f'_{cam}cam'
            for pass_ in ['final','clean']:
                tosave[f'test1024_{pass_}pass{camstr}'] = sorted(tosave[f'test_{pass_}pass{camstr}'])[:-remove][::stride] # warning, it was not sorted before
            assert len(tosave['test1024_cleanpass'])==1024, "incorrect parsing of pairs in Flying Things"
            tosave[f'test1024_allpass{camstr}'] = tosave[f'test1024_cleanpass{camstr}'] + tosave[f'test1024_finalpass{camstr}']
        return tosave
        
      
class MPISintelDataset(FlowDataset):
    
    def _prepare_data(self):
        self.name = "MPISintel"
        self._set_root()
        assert self.split in [s+'_'+p for s in ['train','test','subval','subtrain'] for p in ['cleanpass','finalpass','allpass']]
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, pairname[0], 'frame_{:04d}.png'.format(pairname[1]))
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, pairname[0], 'frame_{:04d}.png'.format(pairname[1]+1))
        self.pairname_to_flowname = lambda pairname: None if pairname[0].startswith('test/') else osp.join(self.root, pairname[0].replace('/clean/','/flow/').replace('/final/','/flow/'), 'frame_{:04d}.flo'.format(pairname[1]))
        self.pairname_to_str = lambda pairname: osp.join(pairname[0], 'frame_{:04d}'.format(pairname[1]))
        self.load_flow = _read_flo_file
        
    def _build_cache(self):
        trainseqs = sorted(os.listdir(self.root+'training/clean'))
        trainpairs = [ (osp.join('training/clean', s),i) for s in trainseqs for i in range(1, len(os.listdir(self.root+'training/clean/'+s)))]
        subvalseqs = ['temple_2','temple_3']
        subtrainseqs = [s for s in trainseqs if s not in subvalseqs]
        subvalpairs = [ (p,i) for p,i in trainpairs if any(s in p for s in subvalseqs)]
        subtrainpairs = [ (p,i) for p,i in trainpairs if any(s in p for s in subtrainseqs)]
        testseqs = sorted(os.listdir(self.root+'test/clean'))
        testpairs = [ (osp.join('test/clean', s),i) for s in testseqs for i in range(1, len(os.listdir(self.root+'test/clean/'+s)))]
        assert len(trainpairs)==1041 and len(testpairs)==552 and len(subvalpairs)==98 and len(subtrainpairs)==943, "incorrect parsing of pairs in MPI-Sintel"
        tosave = {}
        tosave['train_cleanpass'] = trainpairs
        tosave['test_cleanpass'] = testpairs
        tosave['subval_cleanpass'] = subvalpairs
        tosave['subtrain_cleanpass'] = subtrainpairs         
        for t in ['train','test','subval','subtrain']: 
            tosave[t+'_finalpass'] = [(p.replace('/clean/','/final/'),i) for p,i in tosave[t+'_cleanpass']]
            tosave[t+'_allpass'] = tosave[t+'_cleanpass'] + tosave[t+'_finalpass'] 
        return tosave
        
    def submission_save_pairname(self, pairname, prediction, outdir, _time):
        assert prediction.shape[2]==2
        outfile = os.path.join(outdir, 'submission', self.pairname_to_str(pairname)+'.flo')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writeFlowFile(prediction, outfile)
        
    def finalize_submission(self, outdir):
        assert self.split == 'test_allpass'
        bundle_exe = "/nfs/data/ffs-3d/datasets/StereoFlow/MPI-Sintel/bundler/linux-x64/bundler" # eg <bundle_exe> <path_to_results_for_clean> <path_to_results_for_final> <output/bundled.lzma>
        if os.path.isfile(bundle_exe):
            cmd = f'{bundle_exe} "{outdir}/submission/test/clean/" "{outdir}/submission/test/final" "{outdir}/submission/bundled.lzma"'
            print(cmd)
            os.system(cmd)
            print(f'Done. Submission file at: "{outdir}/submission/bundled.lzma"')
        else:
            print('Could not find bundler executable for submission.')
            print('Please download it and run:')
            print(f'<bundle_exe> "{outdir}/submission/test/clean/" "{outdir}/submission/test/final" "{outdir}/submission/bundled.lzma"')
        
class SpringDataset(FlowDataset):

    def _prepare_data(self):
        self.name = "Spring"
        self._set_root()
        assert self.split in ['train','test','subtrain','subval']
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, pairname[0], pairname[1], 'frame_'+pairname[3], 'frame_{:s}_{:04d}.png'.format(pairname[3], pairname[4]))
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, pairname[0], pairname[1], 'frame_'+pairname[3], 'frame_{:s}_{:04d}.png'.format(pairname[3], pairname[4]+(1 if pairname[2]=='FW' else -1)))
        self.pairname_to_flowname = lambda pairname: None if pairname[0]=='test' else osp.join(self.root, pairname[0], pairname[1], f'flow_{pairname[2]}_{pairname[3]}', f'flow_{pairname[2]}_{pairname[3]}_{pairname[4]:04d}.flo5')
        self.pairname_to_str = lambda pairname: osp.join(pairname[0], pairname[1], f'flow_{pairname[2]}_{pairname[3]}', f'flow_{pairname[2]}_{pairname[3]}_{pairname[4]:04d}')
        self.load_flow = _read_hdf5_flow

    def _build_cache(self):
        # train 
        trainseqs = sorted(os.listdir( osp.join(self.root,'train')))
        trainpairs = []
        for leftright in ['left','right']:
            for fwbw in ['FW','BW']:
                trainpairs += [('train',s,fwbw,leftright,int(f[len(f'flow_{fwbw}_{leftright}_'):-len('.flo5')])) for s in trainseqs for f in sorted(os.listdir(osp.join(self.root,'train',s,f'flow_{fwbw}_{leftright}')))]
        # test 
        testseqs = sorted(os.listdir( osp.join(self.root,'test')))
        testpairs = []
        for leftright in ['left','right']:
            testpairs += [('test',s,'FW',leftright,int(f[len(f'frame_{leftright}_'):-len('.png')])) for s in testseqs for f in sorted(os.listdir(osp.join(self.root,'test',s,f'frame_{leftright}')))[:-1]]
            testpairs += [('test',s,'BW',leftright,int(f[len(f'frame_{leftright}_'):-len('.png')])+1) for s in testseqs for f in sorted(os.listdir(osp.join(self.root,'test',s,f'frame_{leftright}')))[:-1]]
        # subtrain / subval
        subtrainpairs = [p for p in trainpairs if p[1]!='0041']
        subvalpairs = [p for p in trainpairs if p[1]=='0041']
        assert len(trainpairs)==19852 and len(testpairs)==3960 and len(subtrainpairs)==19472 and len(subvalpairs)==380, "incorrect parsing of pairs in Spring"
        tosave = {'train': trainpairs, 'test': testpairs, 'subtrain': subtrainpairs, 'subval': subvalpairs}
        return tosave
        
    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==3
        assert prediction.shape[2]==2
        assert prediction.dtype==np.float32
        outfile = osp.join(outdir, pairname[0], pairname[1], f'flow_{pairname[2]}_{pairname[3]}', f'flow_{pairname[2]}_{pairname[3]}_{pairname[4]:04d}.flo5')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writeFlo5File(prediction, outfile)
        
    def finalize_submission(self, outdir):
        assert self.split=='test'
        exe = "{self.root}/flow_subsampling"
        if os.path.isfile(exe):
            cmd = f'cd "{outdir}/test"; {exe} .'
            print(cmd)
            os.system(cmd)
            print(f'Done. Submission file at {outdir}/test/flow_submission.hdf5')
        else:
            print('Could not find flow_subsampling executable for submission.')
            print('Please download it and run:')
            print(f'cd "{outdir}/test"; <flow_subsampling_exe> .')

        
class Kitti12Dataset(FlowDataset):

    def _prepare_data(self):
        self.name = "Kitti12"
        self._set_root()
        assert self.split in ['train','test']
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, pairname+'_10.png')
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, pairname+'_11.png')
        self.pairname_to_flowname = None if self.split=='test' else lambda pairname: osp.join(self.root, pairname.replace('/colored_0/','/flow_occ/')+'_10.png')
        self.pairname_to_str = lambda pairname: pairname.replace('/colored_0/','/')
        self.load_flow = _read_kitti_flow
        
    def _build_cache(self):
        trainseqs = ["training/colored_0/%06d"%(i) for i in range(194)]
        testseqs = ["testing/colored_0/%06d"%(i) for i in range(195)]
        assert len(trainseqs)==194 and len(testseqs)==195, "incorrect parsing of pairs in Kitti12"
        tosave = {'train': trainseqs, 'test': testseqs}
        return tosave 

    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==3
        assert prediction.shape[2]==2
        outfile = os.path.join(outdir, pairname.split('/')[-1]+'_10.png')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writeFlowKitti(outfile, prediction)

    def finalize_submission(self, outdir):
        assert self.split=='test'
        cmd = f'cd {outdir}/; zip -r "kitti12_flow_results.zip" .'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/kitti12_flow_results.zip')


class Kitti15Dataset(FlowDataset):
    
    def _prepare_data(self):
        self.name = "Kitti15"
        self._set_root()
        assert self.split in ['train','subtrain','subval','test']
        self.pairname_to_img1name = lambda pairname: osp.join(self.root, pairname+'_10.png')
        self.pairname_to_img2name = lambda pairname: osp.join(self.root, pairname+'_11.png')
        self.pairname_to_flowname = None if self.split=='test' else lambda pairname: osp.join(self.root, pairname.replace('/image_2/','/flow_occ/')+'_10.png')
        self.pairname_to_str = lambda pairname: pairname.replace('/image_2/','/')
        self.load_flow = _read_kitti_flow
        
    def _build_cache(self):
        trainseqs = ["training/image_2/%06d"%(i) for i in range(200)]
        subtrainseqs = trainseqs[:-10]
        subvalseqs = trainseqs[-10:]
        testseqs = ["testing/image_2/%06d"%(i) for i in range(200)]
        assert len(trainseqs)==200 and len(subtrainseqs)==190 and len(subvalseqs)==10 and len(testseqs)==200, "incorrect parsing of pairs in Kitti15"
        tosave = {'train': trainseqs, 'subtrain': subtrainseqs, 'subval': subvalseqs, 'test': testseqs}
        return tosave 

    def submission_save_pairname(self, pairname, prediction, outdir, time):
        assert prediction.ndim==3
        assert prediction.shape[2]==2
        outfile = os.path.join(outdir, 'flow', pairname.split('/')[-1]+'_10.png')
        os.makedirs( os.path.dirname(outfile), exist_ok=True)
        writeFlowKitti(outfile, prediction)

    def finalize_submission(self, outdir):
        assert self.split=='test'
        cmd = f'cd {outdir}/; zip -r "kitti15_flow_results.zip" flow'
        print(cmd)
        os.system(cmd)
        print(f'Done. Submission file at {outdir}/kitti15_flow_results.zip')


import cv2
def _read_numpy_flow(filename): 
    return np.load(filename)
    
def _read_pfm_flow(filename):
    f, _ = _read_pfm(filename)
    assert np.all(f[:,:,2]==0.0)
    return np.ascontiguousarray(f[:,:,:2])

TAG_FLOAT = 202021.25 # tag to check the sanity of the file
TAG_STRING = 'PIEH'   # string containing the tag
MIN_WIDTH = 1
MAX_WIDTH = 99999
MIN_HEIGHT = 1
MAX_HEIGHT = 99999
def readFlowFile(filename):
    """
    readFlowFile(<FILENAME>) reads a flow file <FILENAME> into a 2-band np.array.
    if <FILENAME> does not exist, an IOError is raised.
    if <FILENAME> does not finish by '.flo' or the tag, the width, the height or the file's size is illegal, an Expcetion is raised.
    ---- PARAMETERS ----
        filename: string containg the name of the file to read a flow
    ---- OUTPUTS ----
        a np.array of dimension (height x width x 2) containing the flow of type 'float32'
    """
        
    # check filename
    if not filename.endswith(".flo"):
        raise Exception("readFlowFile({:s}): filename must finish with '.flo'".format(filename))
    
    # open the file and read it
    with open(filename,'rb') as f:
        # check tag
        tag = struct.unpack('f',f.read(4))[0]
        if tag != TAG_FLOAT:
            raise Exception("flow_utils.readFlowFile({:s}): wrong tag".format(filename))
        # read dimension
        w,h = struct.unpack('ii',f.read(8))
        if w < MIN_WIDTH or w > MAX_WIDTH:
            raise Exception("flow_utils.readFlowFile({:s}: illegal width {:d}".format(filename,w))
        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            raise Exception("flow_utils.readFlowFile({:s}: illegal height {:d}".format(filename,h))
        flow = np.fromfile(f,'float32')
        if not flow.shape == (h*w*2,):
            raise Exception("flow_utils.readFlowFile({:s}: illegal size of the file".format(filename))
        flow.shape = (h,w,2)
        return flow

def writeFlowFile(flow,filename):
    """
    writeFlowFile(flow,<FILENAME>) write flow to the file <FILENAME>.
    if <FILENAME> does not exist, an IOError is raised.
    if <FILENAME> does not finish with '.flo' or the flow has not 2 bands, an Exception is raised.
    ---- PARAMETERS ----
        flow: np.array of dimension (height x width x 2) containing the flow to write
        filename: string containg the name of the file to write a flow
    """
    
    # check filename
    if not filename.endswith(".flo"):
        raise Exception("flow_utils.writeFlowFile(<flow>,{:s}): filename must finish with '.flo'".format(filename))
    
    if not flow.shape[2:] == (2,):
        raise Exception("flow_utils.writeFlowFile(<flow>,{:s}): <flow> must have 2 bands".format(filename))


    # open the file and write it
    with open(filename,'wb') as f:
        # write TAG
        f.write( TAG_STRING.encode('utf-8') )
        # write dimension
        f.write( struct.pack('ii',flow.shape[1],flow.shape[0]) )
        # write the flow
        
        flow.astype(np.float32).tofile(f)
        
_read_flo_file = readFlowFile

def _read_kitti_flow(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    valid = flow[:, :, 2]>0
    flow = flow[:, :, :2]
    flow = (flow - 2 ** 15) / 64.0
    flow[~valid,0] = np.inf
    flow[~valid,1] = np.inf
    return flow
_read_hd1k_flow = _read_kitti_flow
    
        
def writeFlowKitti(filename, uv):
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])

def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)
    
def _read_hdf5_flow(filename):
    flow = np.asarray(h5py.File(filename)['flow'])
    flow[np.isnan(flow)] = np.inf # make invalid values as +inf
    return flow.astype(np.float32)

# flow visualization
RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6
UNKNOWN_THRESH = 1e9

def colorTest():
    """
    flow_utils.colorTest(): display an example of image showing the color encoding scheme
    """
    import matplotlib.pylab as plt
    truerange = 1
    h,w = 151,151
    trange = truerange*1.04
    s2 = round(h/2)
    x,y = np.meshgrid(range(w),range(h))
    u = x*trange/s2-trange
    v = y*trange/s2-trange
    img = _computeColor(np.concatenate((u[:,:,np.newaxis],v[:,:,np.newaxis]),2)/trange/np.sqrt(2))
    plt.imshow(img)
    plt.axis('off')
    plt.axhline(round(h/2),color='k')
    plt.axvline(round(w/2),color='k')
    
def flowToColor(flow, maxflow=None, maxmaxflow=None, saturate=False):
    """
    flow_utils.flowToColor(flow): return a color code flow field, normalized based on the maximum l2-norm of the flow
    flow_utils.flowToColor(flow,maxflow): return a color code flow field, normalized by maxflow
    ---- PARAMETERS ----
        flow: flow to display of shape (height x width x 2)
        maxflow (default:None): if given, normalize the flow by its value, otherwise by the flow norm
        maxmaxflow (default:None): if given, normalize the flow by the max of its value and the flow norm
    ---- OUTPUT ----
        an np.array of shape (height x width x 3) of type uint8 containing a color code of the flow
    """
    h,w,n = flow.shape
    # check size of flow
    assert n == 2, "flow_utils.flowToColor(flow): flow must have 2 bands"
    # fix unknown flow
    unknown_idx = np.max(np.abs(flow),2)>UNKNOWN_THRESH
    flow[unknown_idx] = 0.0
    # compute max flow if needed
    if maxflow is None:
        maxflow = flowMaxNorm(flow)
    if maxmaxflow is not None:
        maxflow = min(maxmaxflow, maxflow)
    # normalize flow
    eps = np.spacing(1) # minimum positive float value to avoid division by 0
    # compute the flow
    img = _computeColor(flow/(maxflow+eps), saturate=saturate)
    # put black pixels in unknown location
    img[ np.tile( unknown_idx[:,:,np.newaxis],[1,1,3]) ] = 0.0 
    return img

def flowMaxNorm(flow):
    """
    flow_utils.flowMaxNorm(flow): return the maximum of the l2-norm of the given flow
    ---- PARAMETERS ----
        flow: the flow
        
    ---- OUTPUT ----
        a float containing the maximum of the l2-norm of the flow
    """
    return np.max( np.sqrt( np.sum( np.square( flow ) , 2) ) )

def _computeColor(flow, saturate=True):
    """
    flow_utils._computeColor(flow): compute color codes for the flow field flow
    
    ---- PARAMETERS ----
        flow: np.array of dimension (height x width x 2) containing the flow to display
    ---- OUTPUTS ----
        an np.array of dimension (height x width x 3) containing the color conversion of the flow
    """
    # set nan to 0
    nanidx = np.isnan(flow[:,:,0])
    flow[nanidx] = 0.0
    
    # colorwheel
    ncols = RY + YG + GC + CB + BM + MR
    nchans = 3
    colorwheel = np.zeros((ncols,nchans),'uint8')
    col = 0;
    #RY
    colorwheel[:RY,0] = 255
    colorwheel[:RY,1] = [(255*i) // RY for i in range(RY)]
    col += RY
    # YG    
    colorwheel[col:col+YG,0] = [255 - (255*i) // YG for i in range(YG)]
    colorwheel[col:col+YG,1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC,1] = 255
    colorwheel[col:col+GC,2] = [(255*i) // GC for i in range(GC)]
    col += GC
    # CB
    colorwheel[col:col+CB,1] = [255 - (255*i) // CB for i in range(CB)]
    colorwheel[col:col+CB,2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM,0] = [(255*i) // BM for i in range(BM)]
    colorwheel[col:col+BM,2] = 255
    col += BM
    # MR
    colorwheel[col:col+MR,0] = 255
    colorwheel[col:col+MR,2] = [255 - (255*i) // MR for i in range(MR)]

    # compute utility variables
    rad = np.sqrt( np.sum( np.square(flow) , 2) ) # magnitude
    a = np.arctan2( -flow[:,:,1] , -flow[:,:,0]) / np.pi # angle
    fk = (a+1)/2 * (ncols-1) # map [-1,1] to [0,ncols-1]
    k0 = np.floor(fk).astype('int')
    k1 = k0+1
    k1[k1==ncols] = 0
    f = fk-k0

    if not saturate:
        rad = np.minimum(rad,1)

    # compute the image
    img = np.zeros( (flow.shape[0],flow.shape[1],nchans), 'uint8' )
    for i in range(nchans):
        tmp = colorwheel[:,i].astype('float')
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1-rad[idx]*(1-col[idx]) # increase saturation with radius
        col[~idx] *= 0.75 # out of range
        img[:,:,i] = (255*col*(1-nanidx.astype('float'))).astype('uint8')

    return img
    
# flow dataset getter 
    
def get_train_dataset_flow(dataset_str, augmentor=True, crop_size=None):
    dataset_str = dataset_str.replace('(','Dataset(')
    if augmentor:
        dataset_str = dataset_str.replace(')',', augmentor=True)')
    if crop_size is not None:
        dataset_str = dataset_str.replace(')',', crop_size={:s})'.format(str(crop_size)))
    return eval(dataset_str)
    
def get_test_datasets_flow(dataset_str):
    dataset_str = dataset_str.replace('(','Dataset(')
    return [eval(s) for s in dataset_str.split('+')]