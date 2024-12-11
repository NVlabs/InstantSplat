# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
from torch.utils.data import Dataset
from PIL import Image

from datasets.transforms import get_pair_transforms

def load_image(impath):
    return Image.open(impath)

def load_pairs_from_cache_file(fname, root=''):
    assert os.path.isfile(fname), "cannot parse pairs from {:s}, file does not exist".format(fname)
    with open(fname, 'r') as fid:
        lines = fid.read().strip().splitlines()
    pairs = [ (os.path.join(root,l.split()[0]), os.path.join(root,l.split()[1])) for l in lines]
    return pairs
    
def load_pairs_from_list_file(fname, root=''):
    assert os.path.isfile(fname), "cannot parse pairs from {:s}, file does not exist".format(fname)
    with open(fname, 'r') as fid:
        lines = fid.read().strip().splitlines()
    pairs = [ (os.path.join(root,l+'_1.jpg'), os.path.join(root,l+'_2.jpg')) for l in lines if not l.startswith('#')]
    return pairs
    
    
def write_cache_file(fname, pairs, root=''):
    if len(root)>0:
        if not root.endswith('/'): root+='/'
        assert os.path.isdir(root)
    s = ''
    for im1, im2 in pairs:
        if len(root)>0:
            assert im1.startswith(root), im1
            assert im2.startswith(root), im2
        s += '{:s} {:s}\n'.format(im1[len(root):], im2[len(root):])
    with open(fname, 'w') as fid:
        fid.write(s[:-1])
    
def parse_and_cache_all_pairs(dname, data_dir='./data/'):
    if dname=='habitat_release':
        dirname = os.path.join(data_dir, 'habitat_release')
        assert os.path.isdir(dirname), "cannot find folder for habitat_release pairs: "+dirname
        cache_file = os.path.join(dirname, 'pairs.txt')
        assert not os.path.isfile(cache_file), "cache file already exists: "+cache_file
        
        print('Parsing pairs for dataset: '+dname)
        pairs = []
        for root, dirs, files in os.walk(dirname):
            if 'val' in root: continue
            dirs.sort()
            pairs += [ (os.path.join(root,f), os.path.join(root,f[:-len('_1.jpeg')]+'_2.jpeg')) for f in sorted(files) if f.endswith('_1.jpeg')]
        print('Found {:,} pairs'.format(len(pairs)))
        print('Writing cache to: '+cache_file)
        write_cache_file(cache_file, pairs, root=dirname)

    else:
        raise NotImplementedError('Unknown dataset: '+dname)
    
def dnames_to_image_pairs(dnames, data_dir='./data/'):
    """
    dnames: list of datasets with image pairs, separated by +
    """
    all_pairs = []
    for dname in dnames.split('+'):
        if dname=='habitat_release':
            dirname = os.path.join(data_dir, 'habitat_release')
            assert os.path.isdir(dirname), "cannot find folder for habitat_release pairs: "+dirname
            cache_file = os.path.join(dirname, 'pairs.txt')
            assert os.path.isfile(cache_file), "cannot find cache file for habitat_release pairs, please first create the cache file, see instructions. "+cache_file
            pairs = load_pairs_from_cache_file(cache_file, root=dirname)
        elif dname in ['ARKitScenes', 'MegaDepth', '3DStreetView', 'IndoorVL']:
            dirname = os.path.join(data_dir, dname+'_crops')
            assert os.path.isdir(dirname), "cannot find folder for {:s} pairs: {:s}".format(dname, dirname)
            list_file = os.path.join(dirname, 'listing.txt')
            assert os.path.isfile(list_file), "cannot find list file for {:s} pairs, see instructions. {:s}".format(dname, list_file)
            pairs = load_pairs_from_list_file(list_file, root=dirname)            
        print('  {:s}: {:,} pairs'.format(dname, len(pairs)))
        all_pairs += pairs 
    if '+' in dnames: print(' Total: {:,} pairs'.format(len(all_pairs)))
    return all_pairs 


class PairsDataset(Dataset):

    def __init__(self, dnames, trfs='', totensor=True, normalize=True, data_dir='./data/'):
        super().__init__()
        self.image_pairs = dnames_to_image_pairs(dnames, data_dir=data_dir)
        self.transforms = get_pair_transforms(transform_str=trfs, totensor=totensor, normalize=normalize)
              
    def __len__(self):
        return len(self.image_pairs)
            
    def __getitem__(self, index):
        im1path, im2path = self.image_pairs[index]
        im1 = load_image(im1path)
        im2 = load_image(im2path)
        if self.transforms is not None: im1, im2 = self.transforms(im1, im2)
        return im1, im2

        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="Computing and caching list of pairs for a given dataset")
    parser.add_argument('--data_dir', default='./data/', type=str, help="path where data are stored")
    parser.add_argument('--dataset', default='habitat_release', type=str, help="name of the dataset")
    args = parser.parse_args()
    parse_and_cache_all_pairs(dname=args.dataset, data_dir=args.data_dir)
