# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from models.croco import CroCoNet
from PIL import Image
import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize, Compose

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count()>0 else 'cpu')
    
    # load 224x224 images and transform them to tensor 
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_mean_tensor = torch.tensor(imagenet_mean).view(1,3,1,1).to(device, non_blocking=True)
    imagenet_std = [0.229, 0.224, 0.225]
    imagenet_std_tensor = torch.tensor(imagenet_std).view(1,3,1,1).to(device, non_blocking=True)
    trfs = Compose([ToTensor(), Normalize(mean=imagenet_mean, std=imagenet_std)])
    image1 = trfs(Image.open('assets/Chateau1.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    image2 = trfs(Image.open('assets/Chateau2.png').convert('RGB')).to(device, non_blocking=True).unsqueeze(0)
    
    # load model 
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    model = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
    model.eval()
    msg = model.load_state_dict(ckpt['model'], strict=True)
    
    # forward 
    with torch.inference_mode():
        out, mask, target = model(image1, image2)
        
    # the output is normalized, thus use the mean/std of the actual image to go back to RGB space 
    patchified = model.patchify(image1)
    mean = patchified.mean(dim=-1, keepdim=True)
    var = patchified.var(dim=-1, keepdim=True)
    decoded_image = model.unpatchify(out * (var + 1.e-6)**.5 + mean)
    # undo imagenet normalization, prepare masked image
    decoded_image = decoded_image * imagenet_std_tensor + imagenet_mean_tensor
    input_image = image1 * imagenet_std_tensor + imagenet_mean_tensor
    ref_image = image2 * imagenet_std_tensor + imagenet_mean_tensor
    image_masks = model.unpatchify(model.patchify(torch.ones_like(ref_image)) * mask[:,:,None])
    masked_input_image = ((1 - image_masks) * input_image)

    # make visualization
    visualization = torch.cat((ref_image, masked_input_image, decoded_image, input_image), dim=3) # 4*(B, 3, H, W) -> B, 3, H, W*4
    B, C, H, W = visualization.shape
    visualization = visualization.permute(1, 0, 2, 3).reshape(C, B*H, W)
    visualization = torchvision.transforms.functional.to_pil_image(torch.clamp(visualization, 0, 1))
    fname = "demo_output.png"
    visualization.save(fname)
    print('Visualization save in '+fname)
    

if __name__=="__main__":
    main()
