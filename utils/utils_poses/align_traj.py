import numpy as np
import torch

from utils.utils_poses.ATE.align_utils import alignTrajectory
from utils.utils_poses.lie_group_helper import SO3_to_quat, convert3x4_4x4


def pts_dist_max(pts):
    """
    :param pts:  (N, 3) torch or np
    :return:     scalar
    """
    if torch.is_tensor(pts):
        dist = pts.unsqueeze(0) - pts.unsqueeze(1)  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = dist.norm(dim=1)  # (N, )
        max_dist = dist.max()
    else:
        dist = pts[None, :, :] - pts[:, None, :]  # (1, N, 3) - (N, 1, 3) -> (N, N, 3)
        dist = dist[0]  # (N, 3)
        dist = np.linalg.norm(dist, axis=1)  # (N, )
        max_dist = dist.max()
    return max_dist


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None, method='sim3'):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)
    quat_a = SO3_to_quat(R_a)  # (N0, 4)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)
    quat_b = SO3_to_quat(R_b)  # (N0, 4)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method=method)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned  # (N1, 4, 4)



def align_scale_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    '''Scale c to b using the scale from a to b.
    :param traj_a:      (N0, 3/4, 4) torch tensor
    :param traj_b:      (N0, 3/4, 4) torch tensor
    :param traj_c:      None or (N1, 3/4, 4) torch tensor
    :return:
        scaled_traj_c   (N1, 4, 4)   torch tensor
        scale           scalar
    '''
    if traj_c is None:
        traj_c = traj_a.clone()

    t_a = traj_a[:, :3, 3]  # (N, 3)
    t_b = traj_b[:, :3, 3]  # (N, 3)

    # scale estimated poses to colmap scale
    # s_a2b: a*s ~ b
    scale_a2b = pts_dist_max(t_b) / pts_dist_max(t_a)

    traj_c[:, :3, 3] *= scale_a2b

    if traj_c.shape[1] == 3:
        traj_c = convert3x4_4x4(traj_c)  # (N, 4, 4)

    return traj_c, scale_a2b  # (N, 4, 4)
