import os
import shutil
import torch
import numpy as np
import argparse
import time
import roma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  (compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images, 
                                 round_python3, rigid_points_registration)

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
    # train_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/train/images")
    # test_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/test/images")
    all_img_folder = os.path.join(img_base_path, f"dust3r_{n_views}_views/init_test_pose/images")
    os.makedirs(all_img_folder, exist_ok=True)
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    #################################################################################################################################################



    # ---------------- (1) Prepare Train & Test images list ---------------- 
    all_img_list = sorted(os.listdir(os.path.join(img_base_path, "images")))
    if args.llffhold > 0:
        train_img_list = [c for idx, c in enumerate(all_img_list) if (idx+1) % args.llffhold != 0]
        test_img_list = [c for idx, c in enumerate(all_img_list) if (idx+1) % args.llffhold == 0]
    # sample sparse view
    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    print(indices)
    tmp_img_list = [train_img_list[i] for i in indices]
    train_img_list = tmp_img_list    
    assert len(train_img_list)==n_views, f"Number of images in the folder is not equal to {n_views}"

    #---------------- (2) Load train pointcloud and intrinsic (define as m1) ---------------- 
    train_pts_all_path = os.path.join(img_base_path, f"dust3r_{n_views}_views", "sparse/0", "pts_4_3dgs_all.npy")
    train_pts_all = np.load(train_pts_all_path)
    train_pts3d_m1 = train_pts_all
    
    if args.focal_avg:
        focal_path = os.path.join(img_base_path, f"dust3r_{n_views}_views", "sparse/0", "focal.npy")
        preset_focal = np.load(focal_path)     # load focal calculated by dust3r_coarse_geometry_initialization



    #---------------- (3) Get N_views pointcloud and 12 test pose (define as n1) ---------------- 
    if len(os.listdir(all_img_folder)) != len(test_img_list):
        for img_name in test_img_list:
            src_path = os.path.join(img_base_path, "images", img_name)
            tgt_path = os.path.join(all_img_folder, "1_"+img_name)
            print(src_path, tgt_path)
            shutil.copy(src_path, tgt_path)
    if len(os.listdir(all_img_folder)) != len(train_img_list):
        for img_name in train_img_list:
            src_path = os.path.join(img_base_path, "images", img_name)
            tgt_path = os.path.join(all_img_folder, "0_"+img_name)
            print(src_path, tgt_path)
            shutil.copy(src_path, tgt_path)
    
    # # read all images
    # all_img_folder = os.path.join(img_base_path, "images")
    images, ori_size = load_images(all_img_folder, size=512)
    print("ori_size", ori_size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)    
    output = inference(pairs, model, args.device, batch_size=batch_size)
    test_output_colmap_path=all_img_folder.replace("images", "sparse/0")
    os.makedirs(test_output_colmap_path, exist_ok=True)

    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg, known_focal=preset_focal[0][0])   
    all_poses = to_numpy(scene.get_im_poses())
    all_pts3d = to_numpy(scene.get_pts3d())

    # train_poses_n1 = [c for idx, c in enumerate(all_poses) if (idx+1) % args.llffhold != 0]    
    # train_pts3d_n1 = [c for idx, c in enumerate(all_pts3d) if (idx+1) % args.llffhold != 0]        
    # train_pts3d_n1 = [c for idx, c in enumerate(all_pts3d) if (idx+1) % args.llffhold != 0]     
    train_pts3d_n1 = all_pts3d[:n_views] 


    # test_poses_n1 = [c for idx, c in enumerate(all_poses) if (idx+1) % args.llffhold == 0]
    test_poses_n1 = all_poses[n_views:] 

    # all_poses_n1 =  np.array(to_numpy(all_poses))
    # train_poses_n1 =  np.array(to_numpy(train_poses_n1))
    train_pts3d_n1 = np.array(to_numpy(train_pts3d_n1)).reshape(-1,3)
    test_poses_n1  = np.array(to_numpy(test_poses_n1))              # test_pose_n1: c2w



    #---------------- (4) Applying pointcloud registration & Calculate transform_matrix & Save initial_test_pose---------------- ##########
    # compute transform that goes from cam to world
    train_pts3d_n1 = torch.from_numpy(train_pts3d_n1)
    train_pts3d_m1 = torch.from_numpy(train_pts3d_m1)
    s, R, T = rigid_points_registration(train_pts3d_n1, train_pts3d_m1)

    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T
    transform_matrix[:3, 3] *= s
    transform_matrix = transform_matrix.numpy()

    test_poses_m1 = transform_matrix @ test_poses_n1
    save_colmap_images(test_poses_m1, os.path.join(test_output_colmap_path, 'images.txt'), test_img_list)


    


  