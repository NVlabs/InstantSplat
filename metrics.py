#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

import scipy
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from utils.utils_poses.relative_pose import compute_relative_world_to_camera
from utils.utils_poses.vis_pose_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def read_colmap_gt_pose(gt_pose_path, llffhold=2):
    # colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/triangulated/images.bin')
    colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/sparse/0/images.bin')
    train_pose=[]
    print("Loading colmap gt train pose:")
    for idx, key in enumerate(colmap_cam_extrinsics):
        if idx % llffhold == 0:
            extr = colmap_cam_extrinsics[key]
            print(idx, extr.name)
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            pose = np.eye(4,4)
            pose[:3, :3] = R
            pose[:3, 3] = T
            train_pose.append(pose)
    colmap_pose = np.array(train_pose)
    return colmap_pose

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s

    return mtx1, mtx2, R

def evaluate(args):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in args.model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                # method_dir = Path(scene_dir) / 'point_cloud' / method
                method_dir = test_dir / method
                out_f = open(method_dir / 'metrics.txt', 'w') 
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    s=ssim(renders[idx], gts[idx])
                    p=psnr(renders[idx], gts[idx])
                    l=lpips(renders[idx], gts[idx], net_type='vgg')
                    out_f.write(f"image name{image_names[idx]}, image idx: {idx}, PSNR: {p.item():.2f}, SSIM: {s:.4f}, LPIPS: {l.item():.4f}\n")
                    ssims.append(s)
                    psnrs.append(p)
                    lpipss.append(l)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


            
            ##### -------  pose_metric ------- #####
            pose_path = Path(scene_dir) / 'pose'
            pose_ours = np.load(pose_path / f'pose_{args.iteration}.npy')
            pose_colmap = read_colmap_gt_pose(args.gt_pose_path)

            # sample sparse view
            indices = np.linspace(0, pose_colmap.shape[0] - 1, args.n_views, dtype=int)
            print("\nCalculating pose metric, train_pose_idx: ", indices)
            tmp_pose_colmap = [pose_colmap[i] for i in indices]
            pose_colmap = tmp_pose_colmap
            
            
            # start to align
            pose_ours = torch.from_numpy(pose_ours)
            poses_gt = np.array(pose_colmap)
            pose_list = []
            for i in range(poses_gt.shape[0]):
                R = poses_gt[i][:3 ,:3].transpose()
                T = poses_gt[i][:3 ,3]
                Rt = np.eye(4, 4)
                Rt[:3, :3] = R
                Rt[:3, 3] = T
                pose_list.append(Rt)
            pose = np.array(pose_list)
            poses_gt = torch.from_numpy(pose)

             # align scale first
            trans_gt_align, trans_est_align, _ = align_pose(poses_gt[:, :3, -1].numpy(),
                                                                pose_ours[:, :3, -1].numpy())
            poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
            pose_ours[:, :3, -1] = torch.from_numpy(trans_est_align)

            c2ws_est_aligned = align_ate_c2b_use_a2b(pose_ours, poses_gt)
            ate = compute_ATE(poses_gt.cpu().numpy(),
                            c2ws_est_aligned.cpu().numpy())
            rpe_trans, rpe_rot = compute_rpe(
                poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
            print("\n")
            print(   
                "RPE_trans: {0:.3f}".format(rpe_trans*100),
                '& RPE_rot: ' "{0:.3f}".format(rpe_rot * 180 / np.pi),
                '& ATE: ', "{0:.3f}".format(ate))
            print("\n")
            
            plot_pose(poses_gt, c2ws_est_aligned, pose_path, args)
            with open(pose_path / f"pose_eval.txt", 'w') as f:
                f.write("RPE_trans: {:.04f}, RPE_rot: {:.04f}, ATE: {:.04f}".format(
                    rpe_trans*100,
                    rpe_rot * 180 / np.pi,
                    ate))
                f.close()
            
            

        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gt_pose_path', type=str, default=None)
    parser.add_argument('--iteration', type=int, default=1000)    
    parser.add_argument("--n_views", default=None, type=int)
    args = parser.parse_args()
    evaluate(args)
