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

import os
import json
from os import makedirs
from time import time, perf_counter
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm
import imageio
import numpy as np
from pathlib import Path

from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.pose_utils import get_tensor_from_camera
from utils.loss_utils import l1_loss, ssim, l1_loss_mask, ssim_loss_mask
from utils.sfm_utils import save_time
from utils.camera_utils import generate_interpolated_path
from utils.camera_utils import visualizer
from arguments import ModelParams, PipelineParams, get_combined_args

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iter}/poses_optimized.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/ours_{iter}/poses_interpolated.png")
    np.save(model_path / f"pose/ours_{iter}/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        if name != "interp":
            torchvision.utils.save_image(   
                gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            )

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=num_iter, eta_min=0.0001)
        with tqdm(total=num_iter, desc=f"Tracking Time Step: {idx+1}", leave=True) as progress_bar:
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            initial_loss = None

            for iteration in range(num_iter):
                rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = l1_loss_mask(rendering, gt, mask)
                loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if iteration == 0:
                        initial_loss = loss.item()  # Capture initial loss

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), initial_loss=initial_loss)
                scheduler.step()

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T

        optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        # print("optimal_pose-camera_pose: ", optimal_pose-camera_pose)
        rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]
            
        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, view.image_name + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, view.image_name + ".png")
        )

    if args.test_fps:
        print(">>> Calculate FPS: ")
        fps_list = []
        for _ in range(1000):
            start = perf_counter()
            _ = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)
            end = perf_counter()
            fps_list.append(end - start)        
        fps_list.sort()
        fps_list = fps_list[100:900]
        fps = 1 / (sum(fps_list) / len(fps_list))
        print(">>> FPS = ", fps)
        with open(f"{model_path}/total_fps.json", 'a') as fp:
            json.dump(f'{fps}', fp, indent=True)
            fp.write('\n')


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # if not skip_train:
    if not skip_train and not args.infer_video and not dataset.eval:        
        optimized_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_optimized.npy')
        viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
        render_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )

    else:
        start_time = time()
        if not skip_test:
            render_set_optimize(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
        end_time = time()
        save_time(dataset.model_path, '[4] render', end_time - start_time)

    if args.infer_video and not dataset.eval:
        save_interpolate_pose(Path(args.model_path), iteration, args.n_views)
        interp_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_interpolated.npy')
        viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
        render_set(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'interp/ours_{iteration}/interp_{args.n_views}_view.mp4')
        images_to_video(image_folder, output_video_file, fps=30)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")    
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--test_fps", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args)
