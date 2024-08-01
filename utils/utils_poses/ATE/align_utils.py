#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np

import utils.utils_poses.ATE.transformations as tfs
import utils.utils_poses.ATE.align_trajectory as align


def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


def alignPositionYawSingle(p_es, p_gt, q_es, q_gt):
    '''
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    '''

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    C_R = np.dot(est_rot, g_rot.transpose())
    theta = align.get_best_yaw(C_R)
    R = align.rot_z(theta)
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
    if n_aligned == 1:
        R, t = alignPositionYawSingle(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        _, R, t = align.align_umeyama(gt_pos, est_pos, known_scale=True,
                                      yaw_only=True)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# align by a SE3 transformation
def alignSE3Single(p_es, p_gt, q_es, q_gt):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    '''

    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]

    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]

    R = np.dot(g_rot, np.transpose(est_rot))
    t = p_gt_0 - np.dot(R, p_es_0)

    return R, t


def alignSE3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    '''
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    '''
    if n_aligned == 1:
        R, t = alignSE3Single(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        s, R, t = align.align_umeyama(gt_pos, est_pos,
                                      known_scale=True)  # note the order
        t = np.array(t)
        t = t.reshape((3, ))
        R = np.array(R)
        return R, t


# align by similarity transformation
def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    '''
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align.align_umeyama(gt_pos, est_pos)  # note the order
    return s, R, t


# a general interface
def alignTrajectory(p_es, p_gt, q_es, q_gt, method, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    '''
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4

    s = 1
    R = None
    t = None
    if method == 'sim3':
        assert n_aligned >= 2 or n_aligned == -1, "sim3 uses at least 2 frames"
        s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'se3':
        R, t = alignSE3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'posyaw':
        R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'none':
        R = np.identity(3)
        t = np.zeros((3, ))
    else:
        assert False, 'unknown alignment method'

    return s, R, t


if __name__ == '__main__':
    pass
