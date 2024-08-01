# This file is modified from NeRF++: https://github.com/Kai-46/nerfplusplus

import numpy as np

try:
    import open3d as o3d
except ImportError:
    pass


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def get_camera_frustum_opengl_coord(H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0., 1., 0.])):
    '''X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    '''
    hfov = np.rad2deg(np.arctan(W / 2. / fx) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / fy) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array([[0., 0., 0., 1.0],                          # frustum origin
                               [-half_w, half_h,  -frustum_length, 1.0],   # top-left image corner
                               [half_w, half_h,   -frustum_length, 1.0],   # top-right image corner
                               [half_w, -half_h,  -frustum_length, 1.0],   # bottom-right image corner
                               [-half_w, -half_h, -frustum_length, 1.0]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])  # (8, 2)
    frustum_colors = np.tile(color.reshape((1, 3)), (frustum_lines.shape[0], 1))  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors

def get_camera_frustum_opencv_coord(H, W, fx, fy, W2C, frustum_length=0.5, color=np.array([0., 1., 0.])):
    '''X right, Y up, Z backward to the observer.
    :param H, W:
    :param fx, fy:
    :param W2C:             (4, 4)  matrix
    :param frustum_length:  scalar: scale the frustum
    :param color:           (3,)    list, frustum line color
    :return:
        frustum_points:     (5, 3)  frustum points in world coordinate
        frustum_lines:      (8, 2)  8 lines connect 5 frustum points, specified in line start/end index.
        frustum_colors:     (8, 3)  colors for 8 lines.
    '''
    hfov = np.rad2deg(np.arctan(W / 2. / fx) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / fy) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum in camera space in homogenous coordinate (5, 4)
    frustum_points = np.array([[0., 0., 0., 1.0],                          # frustum origin
                               [-half_w, -half_h, frustum_length, 1.0],   # top-left image corner
                               [ half_w, -half_h, frustum_length, 1.0],   # top-right image corner
                               [ half_w,  half_h, frustum_length, 1.0],   # bottom-right image corner
                               [-half_w, +half_h, frustum_length, 1.0]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])  # (8, 2)
    frustum_colors = np.tile(color.reshape((1, 3)), (frustum_lines.shape[0], 1))  # (8, 3)

    # transform view frustum from camera space to world space
    C2W = np.linalg.inv(W2C)
    frustum_points = np.matmul(C2W, frustum_points.T).T  # (5, 4)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (5, 3)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors



def draw_camera_frustum_geometry(c2ws, H, W, fx=600.0, fy=600.0, frustum_length=0.5,
                                 color=np.array([29.0, 53.0, 87.0])/255.0, draw_now=False, coord='opengl'):
    '''
    :param c2ws:            (N, 4, 4)  np.array
    :param H:               scalar
    :param W:               scalar
    :param fx:              scalar
    :param fy:              scalar
    :param frustum_length:  scalar
    :param color:           None or (N, 3) or (3, ) or (1, 3) or (3, 1) np array
    :param draw_now:        True/False call o3d vis now
    :return:
    '''
    N = c2ws.shape[0]

    num_ele = color.flatten().shape[0]
    if num_ele == 3:
        color = color.reshape(1, 3)
        color = np.tile(color, (N, 1))

    frustum_list = []
    if coord == 'opengl':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opengl_coord(H, W, fx, fy,
                                                                W2C=np.linalg.inv(c2ws[i]),
                                                                frustum_length=frustum_length,
                                                                color=color[i]))
    elif coord == 'opencv':
        for i in range(N):
            frustum_list.append(get_camera_frustum_opencv_coord(H, W, fx, fy,
                                                                W2C=np.linalg.inv(c2ws[i]),
                                                                frustum_length=frustum_length,
                                                                color=color[i]))
    else:
        print('Undefined coordinate system. Exit')
        exit()

    frustums_geometry = frustums2lineset(frustum_list)

    if draw_now:
        o3d.visualization.draw_geometries([frustums_geometry])

    return frustums_geometry  # this is an o3d geometry object.
