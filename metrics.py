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

# Standard imports
from pathlib import Path
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

# Metrics and image processing
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr

# Pose and SfM utilities
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from utils.utils_poses.vis_pose_utils import plot_pose
from utils.sfm_utils import split_train_test, readImages, align_pose, read_colmap_gt_pose


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

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                
                # ------------------------------ (2) pose evaluation ------------------------------ #
                # load GT Colmap poses
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