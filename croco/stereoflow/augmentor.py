# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Data augmentation for training stereo and flow
# --------------------------------------------------------

# References
# https://github.com/autonomousvision/unimatch/blob/master/dataloader/stereo/transforms.py
# https://github.com/autonomousvision/unimatch/blob/master/dataloader/flow/transforms.py


import numpy as np
import random
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as FF

class StereoAugmentor(object):

    def __init__(self, crop_size, scale_prob=0.5, scale_xonly=True, lhth=800., lminscale=0.0, lmaxscale=1.0, hminscale=-0.2, hmaxscale=0.4, scale_interp_nearest=True, rightjitterprob=0.5, v_flip_prob=0.5, color_aug_asym=True, color_choice_prob=0.5):
        self.crop_size = crop_size
        self.scale_prob = scale_prob
        self.scale_xonly = scale_xonly
        self.lhth = lhth
        self.lminscale = lminscale
        self.lmaxscale = lmaxscale
        self.hminscale = hminscale
        self.hmaxscale = hmaxscale
        self.scale_interp_nearest = scale_interp_nearest
        self.rightjitterprob = rightjitterprob
        self.v_flip_prob = v_flip_prob
        self.color_aug_asym = color_aug_asym
        self.color_choice_prob = color_choice_prob
        
    def _random_scale(self, img1, img2, disp):
        ch,cw = self.crop_size
        h,w = img1.shape[:2]
        if self.scale_prob>0. and np.random.rand()<self.scale_prob:
            min_scale, max_scale = (self.lminscale,self.lmaxscale) if min(h,w) < self.lhth else (self.hminscale,self.hmaxscale)
            scale_x = 2. ** np.random.uniform(min_scale, max_scale)
            scale_x = np.clip(scale_x, (cw+8) / float(w), None)
            scale_y = 1.
            if not self.scale_xonly:
                scale_y = scale_x
                scale_y = np.clip(scale_y, (ch+8) / float(h), None)
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR if not self.scale_interp_nearest else cv2.INTER_NEAREST) * scale_x
        else: # check if we need to resize to be able to crop 
            h,w = img1.shape[:2]
            clip_scale = (cw+8) / float(w)
            if clip_scale>1.:
                scale_x = clip_scale
                scale_y = scale_x if not self.scale_xonly else 1.0
                img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                disp = cv2.resize(disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR if not self.scale_interp_nearest else cv2.INTER_NEAREST) * scale_x
        return img1, img2, disp 
                
    def _random_crop(self, img1, img2, disp): 
        h,w = img1.shape[:2]
        ch,cw = self.crop_size
        assert ch<=h and cw<=w, (img1.shape, h,w,ch,cw)
        offset_x = np.random.randint(w - cw + 1)
        offset_y = np.random.randint(h - ch + 1)
        img1 = img1[offset_y:offset_y+ch,offset_x:offset_x+cw]
        img2 = img2[offset_y:offset_y+ch,offset_x:offset_x+cw]
        disp = disp[offset_y:offset_y+ch,offset_x:offset_x+cw]
        return img1, img2, disp
    
    def _random_vflip(self, img1, img2, disp):
        # vertical flip
        if self.v_flip_prob>0 and np.random.rand() < self.v_flip_prob:
            img1 = np.copy(np.flipud(img1))
            img2 = np.copy(np.flipud(img2))
            disp = np.copy(np.flipud(disp))
        return img1, img2, disp
        
    def _random_rotate_shift_right(self, img2):
        if self.rightjitterprob>0. and np.random.rand()<self.rightjitterprob:
            angle, pixel = 0.1, 2
            px = np.random.uniform(-pixel, pixel)
            ag = np.random.uniform(-angle, angle)
            image_center = (np.random.uniform(0, img2.shape[0]), np.random.uniform(0, img2.shape[1])  )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            img2 = cv2.warpAffine(img2, rot_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR)
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            img2 = cv2.warpAffine(img2, trans_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR)
        return img2
            
    def _random_color_contrast(self, img1, img2):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            img1 = FF.adjust_contrast(img1, contrast_factor)
            if self.color_aug_asym and np.random.random() < 0.5: contrast_factor = np.random.uniform(0.8, 1.2)
            img2 = FF.adjust_contrast(img2, contrast_factor)
        return img1, img2
    def _random_color_gamma(self, img1, img2):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)
            img1 = FF.adjust_gamma(img1, gamma)
            if self.color_aug_asym and np.random.random() < 0.5: gamma = np.random.uniform(0.7, 1.5)
            img2 = FF.adjust_gamma(img2, gamma)
        return img1, img2
    def _random_color_brightness(self, img1, img2):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)
            img1 = FF.adjust_brightness(img1, brightness)
            if self.color_aug_asym and np.random.random() < 0.5: brightness = np.random.uniform(0.5, 2.0)
            img2 = FF.adjust_brightness(img2, brightness)
        return img1, img2
    def _random_color_hue(self, img1, img2):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)
            img1 = FF.adjust_hue(img1, hue)
            if self.color_aug_asym and np.random.random() < 0.5: hue = np.random.uniform(-0.1, 0.1)
            img2 = FF.adjust_hue(img2, hue)
        return img1, img2
    def _random_color_saturation(self, img1, img2):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)
            img1 = FF.adjust_saturation(img1, saturation)
            if self.color_aug_asym and np.random.random() < 0.5: saturation = np.random.uniform(-0.8,1.2)
            img2 = FF.adjust_saturation(img2, saturation)
        return img1, img2   
    def _random_color(self, img1, img2):
        trfs = [self._random_color_contrast,self._random_color_gamma,self._random_color_brightness,self._random_color_hue,self._random_color_saturation]
        img1 = Image.fromarray(img1.astype('uint8'))
        img2 = Image.fromarray(img2.astype('uint8'))
        if np.random.random() < self.color_choice_prob:
            # A single transform
            t = random.choice(trfs)
            img1, img2 = t(img1, img2)
        else:
            # Combination of trfs
            # Random order
            random.shuffle(trfs)
            for t in trfs:
                img1, img2 = t(img1, img2)
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)
        return img1, img2
                    
    def __call__(self, img1, img2, disp, dataset_name):
        img1, img2, disp = self._random_scale(img1, img2, disp)
        img1, img2, disp = self._random_crop(img1, img2, disp)
        img1, img2, disp = self._random_vflip(img1, img2, disp)
        img2 = self._random_rotate_shift_right(img2)
        img1, img2 = self._random_color(img1, img2)
        return img1, img2, disp



class FlowAugmentor:

    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, spatial_aug_prob=0.8, stretch_prob=0.8, max_stretch=0.2, h_flip_prob=0.5, v_flip_prob=0.1, asymmetric_color_aug_prob=0.2):
    
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = stretch_prob
        self.max_stretch = max_stretch

        # flip augmentation params
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

        self.asymmetric_color_aug_prob = asymmetric_color_aug_prob
        
    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def _resize_flow(self, flow, scale_x, scale_y, factor=1.0):
        if np.all(np.isfinite(flow)):
            flow = cv2.resize(flow, None, fx=scale_x/factor, fy=scale_y/factor, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]
        else: # sparse version
            fx, fy = scale_x, scale_y
            ht, wd = flow.shape[:2]
            coords = np.meshgrid(np.arange(wd), np.arange(ht))
            coords = np.stack(coords, axis=-1)

            coords = coords.reshape(-1, 2).astype(np.float32)
            flow = flow.reshape(-1, 2).astype(np.float32)
            valid = np.isfinite(flow[:,0])

            coords0 = coords[valid]
            flow0 = flow[valid]

            ht1 = int(round(ht * fy/factor))
            wd1 = int(round(wd * fx/factor))
            
            rescale = np.expand_dims(np.array([fx, fy]), axis=0)
            coords1 = coords0 * rescale / factor
            flow1 = flow0 * rescale

            xx = np.round(coords1[:, 0]).astype(np.int32)
            yy = np.round(coords1[:, 1]).astype(np.int32)

            v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
            xx = xx[v]
            yy = yy[v]
            flow1 = flow1[v]

            flow = np.inf * np.ones([ht1, wd1, 2], dtype=np.float32) # invalid value every where, before we fill it with the correct ones
            flow[yy, xx] = flow1
        return flow
        
    def spatial_transform(self, img1, img2, flow, dname):
    
        if np.random.rand() < self.spatial_aug_prob:
            # randomly sample scale
            ht, wd = img1.shape[:2]
            clip_min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(ht),
                (self.crop_size[1] + 8) / float(wd))
            min_scale, max_scale = self.min_scale, self.max_scale
            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale
            if np.random.rand() < self.stretch_prob:
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_x = np.clip(scale_x, clip_min_scale, None)
            scale_y = np.clip(scale_y, clip_min_scale, None)
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = self._resize_flow(flow, scale_x, scale_y, factor=2.0 if dname=='Spring' else 1.0)
        elif dname=="Spring":
            flow = self._resize_flow(flow, 1.0, 1.0, factor=2.0)

        if self.h_flip_prob>0. and np.random.rand() < self.h_flip_prob:  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]

        if self.v_flip_prob>0. and np.random.rand() < self.v_flip_prob:  # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]
                
        # In case no cropping
        if img1.shape[0] - self.crop_size[0] > 0:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        else:
            y0 = 0
        if img1.shape[1] - self.crop_size[1] > 0:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        else:
            x0 = 0

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow, dname):
        img1, img2, flow = self.spatial_transform(img1, img2, flow, dname)
        img1, img2 = self.color_transform(img1, img2)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        return img1, img2, flow