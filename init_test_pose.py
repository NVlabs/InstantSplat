import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks, rigid_points_registration)
from utils.camera_utils import generate_interpolated_path


def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter, 
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False, focal_avg=True, infer_video=False):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # when init test pose, use all images
    image_files = train_img_files + test_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)

    start_time = time()
    print(f'>> Making pairs...')
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f'>> Inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    # Load Focal
    train_pts_all_path = sparse_0_path / 'points3D_all.npy'
    train_pts_all = np.load(train_pts_all_path)
    train_pts3d_m1 = train_pts_all
    if args.focal_avg:
        focals_file = sparse_0_path / 'non_scaled_focals.npy'
        preset_focals = np.load(focals_file)
        preset_focal = np.mean(preset_focals)
        print(f">> preset_focal: {preset_focal}")
        
    print(f'>> Global alignment...')
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=500, schedule=schedule, lr=lr, focal_avg=args.focal_avg, known_focal=preset_focal)    
    imgs = np.array(scene.imgs)
    focals = np.repeat(preset_focal, len(test_img_files))

    all_poses = to_numpy(scene.get_im_poses())
    all_pts3d = to_numpy(scene.get_pts3d())
    train_pts3d_n1 = all_pts3d[:n_views]
    test_poses_n1 = all_poses[n_views:]
    train_pts3d_n1 = np.array(to_numpy(train_pts3d_n1)).reshape(-1,3)
    test_poses_n1  = np.array(to_numpy(test_poses_n1))              # test_pose_n1: c2w

    #---------------- (4) Applying pointcloud registration & Calculate transform_matrix & Save initial_test_pose---------------- 
    # compute transform that goes from cam to world
    train_pts3d_n1 = torch.from_numpy(train_pts3d_n1)
    train_pts3d_m1 = torch.from_numpy(train_pts3d_m1)
    scale, R, T = rigid_points_registration(train_pts3d_n1, train_pts3d_m1, conf=None)

    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    transform_matrix[:3, 3] *= scale
    transform_matrix = transform_matrix.numpy()
    test_poses_m1 = transform_matrix @ test_poses_n1

    # Save results
    print(f'>> Saving results...')
    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[3] init_test_pose', Train_Time)
    save_extrinsic(sparse_1_path, inv(test_poses_m1), test_img_files, image_suffix)
    save_intrinsics(sparse_1_path, focals, org_imgs_shape, imgs.shape, save_focals=False)
    print(f'[INFO] MASt3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_1_path)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--ckpt_path', type=str,
        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='')
    parser.add_argument('--n_views', type=int, default=3, help='')
    # parser.add_argument('--focal_avg', type=bool, default=False, help='')
    parser.add_argument('--focal_avg', action="store_true")
    parser.add_argument('--conf_aware_ranking', action="store_true")
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold')
    parser.add_argument('--infer_video', action="store_true")

    args = parser.parse_args()
    main(args.source_path, args.model_path, args.ckpt_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter,         
          args.min_conf_thr, args.llffhold, args.n_views, args.co_vis_dsp, args.depth_thre, args.conf_aware_ranking, args.focal_avg, args.infer_video)
