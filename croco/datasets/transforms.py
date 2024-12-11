# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torchvision.transforms
import torchvision.transforms.functional as F

# "Pair": apply a transform on a pair
# "Both": apply the exact same transform to both images

class ComposePair(torchvision.transforms.Compose):
    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2

class NormalizeBoth(torchvision.transforms.Normalize):
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2

class ToTensorBoth(torchvision.transforms.ToTensor):
    def __call__(self, img1, img2):
        img1 = super().__call__(img1)
        img2 = super().__call__(img2)
        return img1, img2
        
class RandomCropPair(torchvision.transforms.RandomCrop): 
    # the crop will be intentionally different for the two images with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2

class ColorJitterPair(torchvision.transforms.ColorJitter): 
    # can be symmetric (same for both images) or assymetric (different jitter params for each image) depending on assymetric_prob  
    def __init__(self, assymetric_prob, **kwargs):
        super().__init__(**kwargs)
        self.assymetric_prob = assymetric_prob
    def jitter_one(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img
        
    def forward(self, img1, img2):

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img1 = self.jitter_one(img1, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        if torch.rand(1) < self.assymetric_prob: # assymetric:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        img2 = self.jitter_one(img2, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        return img1, img2

def get_pair_transforms(transform_str, totensor=True, normalize=True):
    # transform_str is eg    crop224+color
    trfs = []
    for s in transform_str.split('+'):
        if s.startswith('crop'):
            size = int(s[len('crop'):])
            trfs.append(RandomCropPair(size))
        elif s=='acolor':
            trfs.append(ColorJitterPair(assymetric_prob=1.0, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.0))
        elif s=='': # if transform_str was ""
            pass
        else:
            raise NotImplementedError('Unknown augmentation: '+s)
            
    if totensor:
        trfs.append( ToTensorBoth() )
    if normalize:
        trfs.append( NormalizeBoth(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) )

    if len(trfs)==0:
        return None
    elif len(trfs)==1:
        return trfs
    else:
        return ComposePair(trfs)
        
        
        
        
        
