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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_torch(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = torch.tensor(translate, dtype=torch.float32)
    
    # Initialize the transformation matrix
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.t()  # Transpose of R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Compute the inverse to get the camera-to-world transformation
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    # Invert again to get the world-to-view transformation
    Rt = torch.linalg.inv(C2W)
    
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def cumulative_sum(input_list):
    cumulative_list = [0]
    current_sum = 0
    for num in input_list:
        current_sum += num
        cumulative_list.append(current_sum)
    return cumulative_list

def compute_scale_gaussian_by_project_pair_pcd(points_3d_all, extrins, intrins, view_num_list=None):
    frame_num = extrins.shape[0]
    per_view_num = points_3d_all.shape[0]//frame_num
    if view_num_list is None:
        select_range = cumulative_sum(np.tile(per_view_num, (frame_num)))
    else:
        select_range = cumulative_sum(view_num_list)

    depth_z = []
    for ii in range(frame_num):
        print(f"view {ii}, points {select_range[ii]} {select_range[ii+1]}")
        points_3d = points_3d_all
        extrin = extrins[ii]
        intrin = intrins[ii]

        R = extrin[:3,:3]
        t = extrin[:3, 3]
        points_cam = R @ points_3d.T + t[:, np.newaxis]  # Broadcasting t for each point

        fx, fy = intrin

        depths = points_cam[2, :]
        depth_z.append(depths)
    depth_z = np.array(depth_z)
    depth_z = np.min(depth_z, 0)
    depth_z = np.clip(depth_z, 0.01, depth_z.max())
    scale_gaussian = depth_z / ((fx + fy)/2)
    print("compute_scale_gaussian_by_project_pair_pcd", points_3d_all.shape, scale_gaussian.shape)
    return scale_gaussian