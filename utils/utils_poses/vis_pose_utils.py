import os
import matplotlib
import matplotlib.pyplot as plt
import copy
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.main_ape import ape
from evo.tools import plot
from evo.core import sync
from evo.tools import file_interface
from evo.core import metrics
import evo
import torch
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as si


def interp_poses(c2ws, N_views):
    N_inputs = c2ws.shape[0]
    trans = c2ws[:, :3, 3:].permute(2, 1, 0)
    rots = c2ws[:, :3, :3]
    render_poses = []
    rots = R.from_matrix(rots)
    slerp = Slerp(np.linspace(0, 1, N_inputs), rots)
    interp_rots = torch.tensor(
        slerp(np.linspace(0, 1, N_views)).as_matrix().astype(np.float32))
    interp_trans = torch.nn.functional.interpolate(
        trans, size=N_views, mode='linear').permute(2, 1, 0)
    render_poses = torch.cat([interp_rots, interp_trans], dim=2)
    render_poses = convert3x4_4x4(render_poses)
    return render_poses


def interp_poses_bspline(c2ws, N_novel_imgs, input_times, degree):
    target_trans = torch.tensor(scipy_bspline(
        c2ws[:, :3, 3], n=N_novel_imgs, degree=degree, periodic=False).astype(np.float32)).unsqueeze(2)
    rots = R.from_matrix(c2ws[:, :3, :3])
    slerp = Slerp(input_times, rots)
    target_times = np.linspace(input_times[0], input_times[-1], N_novel_imgs)
    target_rots = torch.tensor(
        slerp(target_times).as_matrix().astype(np.float32))
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([0.7*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        # c = np.dot(c2w[:3,:4], np.array([0.3*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.01, 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array(
            [0.2*np.cos(theta), -0.2*np.sin(theta), -np.sin(theta*zrate) * 0.1, 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate(
            (cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count-1)
        kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))


def generate_spiral_nerf(learned_poses, bds, N_novel_views, hwf):
    learned_poses_ = np.concatenate((learned_poses[:, :3, :4].detach(
    ).cpu().numpy(), hwf[:len(learned_poses)]), axis=-1)
    c2w = poses_avg(learned_poses_)
    print('recentered', c2w.shape)
    # Get spiral
    # Get average pose
    up = normalize(learned_poses_[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset

    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = learned_poses_[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_rots = 2
    c2ws = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_novel_views)
    c2ws = torch.tensor(np.stack(c2ws).astype(np.float32))
    c2ws = c2ws[:, :3, :4]
    c2ws = convert3x4_4x4(c2ws)
    return c2ws


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(
                input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor(
                [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


plt.rc('legend', fontsize=20)  # using a named size


def plot_pose(ref_poses, est_poses, output_path, args, vid=False):
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)
    if vid:
        for p_idx in range(len(ref_poses)):
            fig = plt.figure()
            current_est_aligned = traj_est_aligned.poses_se3[:p_idx+1]
            current_ref = traj_ref.poses_se3[:p_idx+1]
            current_est_aligned = PosePath3D(poses_se3=current_est_aligned)
            current_ref = PosePath3D(poses_se3=current_ref)
            traj_by_label = {
                # "estimate (not aligned)": traj_est,
                "Ours (aligned)": current_est_aligned,
                "Ground-truth": current_ref
            }
            plot_mode = plot.PlotMode.xyz
            # ax = plot.prepare_axis(fig, plot_mode, 111)
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            colors = ['r', 'b']
            styles = ['-', '--']

            for idx, (label, traj) in enumerate(traj_by_label.items()):
                plot.traj(ax, plot_mode, traj,
                          styles[idx], colors[idx], label)
                # break
            # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            os.makedirs(os.path.join(os.path.dirname(
                output_path), 'pose_vid'), exist_ok=True)
            pose_vis_path = os.path.join(os.path.dirname(
                output_path), 'pose_vid', 'pose_vis_{:03d}.png'.format(p_idx))
            print(pose_vis_path)
            fig.savefig(pose_vis_path)

    # else:

    fig = plt.figure()
    fig.patch.set_facecolor('white')                    # 把背景设置为纯白色
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
    
        "Ours (aligned)": traj_est_aligned,
        # "NoPe-NeRF (aligned)": traj_est_aligned,
        # "CF-3DGS (aligned)": traj_est_aligned,
        # "NeRFmm (aligned)": traj_est_aligned,
        # args.method + " (aligned)": traj_est_aligned,
        "COLMAP (GT)": traj_ref
        # "Ground-truth": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    # ax = plot.prepare_axis(fig, plot_mode, 111)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('white')                           # 把子图设置为纯白色
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.zaxis.set_tick_params(labelleft=True)
    colors = ['#2c9e38', '#d12920']     # 
    # colors = ['#2c9e38', '#a72126']     # 

    # colors = ['r', 'b']
    styles = ['-', '--']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)
        # break
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    ax.view_init(elev=30., azim=45)
    # ax.view_init(elev=10., azim=45)
    plt.tight_layout()
    pose_vis_path = output_path / f'pose_vis.png'
    # pose_vis_path = os.path.join(os.path.dirname(output_path), f'pose_vis_{args.method}_{args.scene}.png')
    fig.savefig(pose_vis_path)

    # path_parts = args.pose_path.split('/')
    # tmp_vis_path = '/'.join(path_parts[:-1]) + '/all_vis'
    # tmp_vis_path2 = os.path.join(tmp_vis_path, f'pose_vis_{args.method}_{args.scene}.png')
    # fig.savefig(tmp_vis_path2)
