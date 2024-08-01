import os
import shutil
import torch
import numpy as np
import argparse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    # parser.add_argument("--model_path", type=str, default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--model_path", type=str, default="submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", action="store_true")
    # parser.add_argument("--focal_avg", type=bool, default=True)

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--img_base_path", type=str, default="/home/workspace/datasets/instantsplat/Tanks_dust3r/Barn/24_views")

    return parser

if __name__ == '__main__':
    
    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path
    img_folder_path = os.path.join(img_base_path, f"dust3r_{n_views}_views/images")
    os.makedirs(img_folder_path, exist_ok=True)
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    ##########################################################################################################################################################################################
    
    train_img_list = sorted(os.listdir(os.path.join(img_base_path, "images")))
    if args.llffhold > 0:
        train_img_list = [c for idx, c in enumerate(train_img_list) if (idx+1) % args.llffhold != 0]
    
    # sample sparse view
    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    print(indices)
    tmp_img_list = [train_img_list[i] for i in indices]
    train_img_list = tmp_img_list
    
    assert len(train_img_list)==n_views, f"Number of images in the folder is not equal to {n_views}"
    
    if len(os.listdir(img_folder_path)) != len(train_img_list):
        for img_name in train_img_list:
            src_path = os.path.join(img_base_path, "images", img_name)
            tgt_path = os.path.join(img_folder_path, img_name)
            print(src_path, tgt_path)
            shutil.copy(src_path, tgt_path)
    images, ori_size = load_images(img_folder_path, size=512) 
    print("ori_size", ori_size)

    start_time = time.time()
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=batch_size)
    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()   

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))
