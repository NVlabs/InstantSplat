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
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b, align_scale_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from utils.utils_poses.relative_pose import compute_relative_world_to_camera
from utils.utils_poses.vis_pose_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.sfm_utils import split_train_test, readImages, align_pose, read_colmap_gt_pose
import tifffile

def calculate_depth_metrics(pred_depth, gt_depth):
    mask = (gt_depth > 0) & (pred_depth > 0)
    gt_depth_masked = gt_depth[mask]
    pred_depth_masked = pred_depth[mask]
    
    if gt_depth_masked.numel() == 0 or pred_depth_masked.numel() == 0:
        return torch.tensor(float('nan')), torch.tensor(float('nan'))
    
    median_gt, median_pred = torch.median(gt_depth_masked), torch.median(pred_depth_masked)
    if torch.isclose(median_pred, torch.tensor(0.0)):
        return torch.tensor(float('nan')), torch.tensor(float('nan'))
    
    pred_depth_masked *= (median_gt / median_pred)  # scale adjustment
    
    rel_err = torch.abs(gt_depth_masked - pred_depth_masked) / gt_depth_masked
    rel_err[torch.isnan(rel_err)] = 0
    
    ratio = torch.max(pred_depth_masked / gt_depth_masked, gt_depth_masked / pred_depth_masked)
    tau = (ratio < 1.03).float().mean()

    return rel_err.mean() * 100, tau * 100
    

def evaluate(args):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in args.model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                # ------------------------------ (1) image evaluation ------------------------------ #
                method_dir = test_dir / method
                out_f = open(method_dir / 'metrics.txt', 'w') 
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                gt_depth_dir = Path(args.source_path) / "depths"
                gt_depth_files = sorted([f.name for f in gt_depth_dir.glob('*_depth.npy')])
                pred_depth_dir = method_dir / "vis"

                ssims = []
                psnrs = []
                lpipss = []
                rel_errs = []
                taus = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    s=ssim(renders[idx], gts[idx])
                    p=psnr(renders[idx], gts[idx])
                    l=lpips(renders[idx], gts[idx], net_type='vgg')
                    out_f.write(f"image name{image_names[idx]}, image idx: {idx}, PSNR: {p.item():.2f}, SSIM: {s:.4f}, LPIPS: {l.item():.4f}\n")
                    ssims.append(s)
                    psnrs.append(p)
                    lpipss.append(l)

                    gt_depth = torch.from_numpy(np.load(gt_depth_dir / gt_depth_files[idx])).squeeze()
                    pred_depth = torch.from_numpy(tifffile.imread(pred_depth_dir / f"depth_{idx:05d}.tiff"))
                    rel_err, tau = calculate_depth_metrics(pred_depth, gt_depth)
                    rel_errs.append(rel_err)
                    taus.append(tau)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  Rel_err: {:>12.7f}".format(torch.tensor(rel_errs).mean(), ".5"))
                print("  Tau: {:>12.7f}".format(torch.tensor(taus).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "Rel_err": torch.tensor(rel_errs).mean().item(),
                                                        "Tau": torch.tensor(taus).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "Rel_err": {name: rel_err for rel_err, name in zip(torch.tensor(rel_errs).tolist(), image_names)},
                                                            "Tau": {name: tau for tau, name in zip(torch.tensor(taus).tolist(), image_names)}})
                
                # ------------------------------ (2) pose evaluation ------------------------------ #
                pose_dir = Path(scene_dir) / "pose"
                pose_path = pose_dir / method
                pose_optimized = np.load(pose_path / f'pose_optimized.npy')
                pose_colmap = read_colmap_gt_pose(args.source_path)

                gt_train_pose, _ = split_train_test(pose_colmap, llffhold=8, n_views=args.n_views, verbose=False)
                # start to align
                pose_optimized = torch.from_numpy(pose_optimized)
                poses_gt = torch.from_numpy(np.array(gt_train_pose))
                # align scale first
                trans_gt_align, trans_est_align, _ = align_pose(poses_gt[:, :3, -1].numpy(), pose_optimized[:, :3, -1].numpy())
                poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
                pose_optimized[:, :3, -1] = torch.from_numpy(trans_est_align)

                c2ws_est_aligned = align_ate_c2b_use_a2b(pose_optimized, poses_gt)
                ate = compute_ATE(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
                rpe_trans, rpe_rot = compute_rpe(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
                print(f"  RPE_t: {rpe_trans*100:>12.7f}")
                print(f"  RPE_r: {rpe_rot * 180 / np.pi:>12.7f}")
                print(f"  ATE  : {ate:>12.7f}")
                print("")
                full_dict[scene_dir][method].update({"RPE_t": rpe_trans*100,
                                                    "RPE_r": rpe_rot * 180 / np.pi,
                                                    "ATE": ate})
                plot_pose(poses_gt, c2ws_est_aligned, pose_path, args)
                with open(pose_path / f"pose_eval.txt", 'w') as f:
                    f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                        rpe_trans*100,
                        rpe_rot * 180 / np.pi,
                        ate))

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_path', '-s', required=True, type=str, default=None)
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--n_views", default=None, type=int)
    args = parser.parse_args()
    evaluate(args)