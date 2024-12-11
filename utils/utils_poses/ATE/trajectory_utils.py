# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import numpy as np
import utils.utils_poses.ATE.transformations as tf


def get_rigid_body_trafo(quat, trans):
    T = tf.quaternion_matrix(quat)
    T[0:3, 3] = trans
    return T


def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:, 0:3], axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances


def compute_comparison_indices_length(distances, dist, max_dist_diff):
    max_idx = len(distances)
    comparisons = []
    for idx, d in enumerate(distances):
        best_idx = -1
        error = max_dist_diff
        for i in range(idx, max_idx):
            if np.abs(distances[i]-(d+dist)) < error:
                best_idx = i
                error = np.abs(distances[i] - (d+dist))
        if best_idx != -1:
            comparisons.append(best_idx)
    return comparisons


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1)/2)))*180.0/np.pi
