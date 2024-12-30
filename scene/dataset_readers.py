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
import sys
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import copy
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_poses: list
    test_poses: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def loadCameras(poses, viewpoint_stack):

    # load optimized poses
    if poses.shape[0] == len(viewpoint_stack):
        for idx, cam in enumerate(viewpoint_stack):
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            cam.R = R
            cam.T = T
            cam.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

    # load interpolated poses
    elif poses.shape[0] > len(viewpoint_stack):
        repeat_times = int(np.ceil(poses.shape[0] / len(viewpoint_stack)))
        # Create repeated list instead of using np.tile
        viewpoint_stack = [copy.deepcopy(vp) for vp in viewpoint_stack * repeat_times][:poses.shape[0]]
        for idx in range(poses.shape[0]):                                 
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            viewpoint_stack[idx].uid = idx           
            viewpoint_stack[idx].colmap_id = idx+1    
            viewpoint_stack[idx].image_name = str(idx).zfill(5)    
            viewpoint_stack[idx].R = R
            viewpoint_stack[idx].T = T            
            viewpoint_stack[idx].world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            viewpoint_stack[idx].full_proj_transform = (viewpoint_stack[idx].world_view_transform.unsqueeze(0).bmm(viewpoint_stack[idx].projection_matrix.unsqueeze(0))).squeeze(0)
            viewpoint_stack[idx].camera_center = viewpoint_stack[idx].world_view_transform.inverse()[3, :3]
    return viewpoint_stack


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    poses=[]
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose = np.block([[R, T.reshape(3, 1)], [np.zeros((1, 3)), 1]])
        poses.append(pose)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            f, cx, cy, r = intr.params
            FovY = focal2fov(f, height)
            FovX = focal2fov(f, width)
            prcppoint = np.array([cx / width, cy / height])
            # undistortion
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            D = np.array([r, 0, 0, 0])  # Only radial distortion
            image_undistorted = cv2.undistort(image_cv, K, D, None)
            image_undistorted = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_undistorted)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, poses



# For interpolated video, open when only render interpolated video
def readColmapCamerasInterp(cam_extrinsics, cam_intrinsics, images_folder, model_path):
    
    pose_interpolated_path = model_path + 'pose/pose_interpolated.npy'
    pose_interpolated = np.load(pose_interpolated_path)
    intr = cam_intrinsics[1]

    cam_infos = []
    poses=[]
    for idx, pose_npy in enumerate(pose_interpolated):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_interpolated.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        # print(uid)
        # print(pose.shape)
        # pose = np.linalg.inv(pose)
        poses.append(pose)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        images_list = os.listdir(os.path.join(images_folder))
        image_name_0 = images_list[0]
        image_name = str(idx).zfill(4)
        image = Image.open(images_folder + '/' + image_name_0)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=images_folder, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, poses


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# def readColmapSceneInfo2(path, images, eval, args, opt, llffhold=8):
#     try:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     except:
#         cameras_extrinsic_file = os.path.join(path, f"sparse/0", "images.txt")
#         cameras_intrinsic_file = os.path.join(path, f"sparse/0", "cameras.txt")
#         cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     reading_dir = "images" if images == None else images

#     cam_infos_unsorted, poses = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
#     sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
#     cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
#     sorted_poses = [poses[i] for i in sorting_indices]
#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

#     # Train-Test Split---CVPR
#     start_index = int(llffhold/2)
#     test_indices     = [idx for idx in range(start_index, len(cam_infos)) if idx % llffhold == 0]
#     non_test_indices = [idx for idx in range(start_index, len(cam_infos)) if idx % llffhold != 0]
#     if opt.n_views is None or opt.n_views == 0:
#         opt.n_views = len(non_test_indices)
#     sparse_indices = np.linspace(0, len(non_test_indices) - 1, opt.n_views, dtype=int)
#     train_indices = [non_test_indices[i] for i in sparse_indices]
#     print(">> Spliting Train-Test Set: ")
#     print(" - sparse_idx:         ", sparse_indices)
#     print(" - train_set_indices:  ", train_indices)
#     print(" - test_set_indices:   ", test_indices)
#     train_cam_infos = [cam_infos[i] for i in train_indices]
#     test_cam_infos = [cam_infos[i] for i in test_indices]

#     raise ValueError("An error occurred")

#     # if eval:
#     #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1) % llffhold != 0]
#     #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if (idx+1) % llffhold == 0]
#     #     train_poses = [c for idx, c in enumerate(sorted_poses) if (idx+1) % llffhold != 0]
#     #     test_poses = [c for idx, c in enumerate(sorted_poses) if (idx+1) % llffhold == 0]

#     # else:
#     #     train_cam_infos = cam_infos
#     #     test_cam_infos = []
#     #     train_poses = sorted_poses
#     #     test_poses = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, f"sparse_{opt.n_views}/0/points3D.ply")
#     bin_path = os.path.join(path, f"sparse_{opt.n_views}/0/points3D.bin")
#     txt_path = os.path.join(path, f"sparse_{opt.n_views}/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             xyz, rgb, _ = read_points3D_text(txt_path)
#         storePly(ply_path, xyz, rgb)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            train_poses=None,
#                            test_poses=None)
#     return scene_info

def readColmapSceneInfo(path, images, eval, args, llffhold=8):

    if eval:
        cameras_extrinsic_file = os.path.join(path, f"sparse_{args.n_views}/1", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"sparse_{args.n_views}/1", "cameras.txt")
    else:
        cameras_extrinsic_file = os.path.join(path, f"sparse_{args.n_views}/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"sparse_{args.n_views}/0", "cameras.txt")

    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images

    cam_infos_unsorted, poses = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
    cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
    sorted_poses = [poses[i] for i in sorting_indices]
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos
        train_poses = sorted_poses
        test_poses = sorted_poses
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        train_poses = sorted_poses
        test_poses = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"sparse_{args.n_views}/0/points3D.ply")
    bin_path = os.path.join(path, f"sparse_{args.n_views}/0/points3D.bin")
    txt_path = os.path.join(path, f"sparse_{args.n_views}/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_poses=train_poses,
                           test_poses=test_poses)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}