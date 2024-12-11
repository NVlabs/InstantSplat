# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import numpy as np
import quaternion
import habitat_sim
import json
from sklearn.neighbors import NearestNeighbors
import cv2

# OpenCV to habitat camera convention transformation
R_OPENCV2HABITAT = np.stack((habitat_sim.geo.RIGHT, -habitat_sim.geo.UP, habitat_sim.geo.FRONT), axis=0)
R_HABITAT2OPENCV = R_OPENCV2HABITAT.T
DEG2RAD = np.pi / 180

def compute_camera_intrinsics(height, width, hfov):
    f = width/2 / np.tan(hfov/2 * np.pi/180)
    cu, cv = width/2, height/2
    return f, cu, cv

def compute_camera_pose_opencv_convention(camera_position, camera_orientation):
    R_cam2world = quaternion.as_rotation_matrix(camera_orientation) @ R_OPENCV2HABITAT
    t_cam2world = np.asarray(camera_position)
    return R_cam2world, t_cam2world

def compute_pointmap(depthmap, hfov):
    """ Compute a HxWx3 pointmap in camera frame from a HxW depth map."""
    height, width = depthmap.shape
    f, cu, cv = compute_camera_intrinsics(height, width, hfov)
    # Cast depth map to point
    z_cam = depthmap
    u, v = np.meshgrid(range(width), range(height))
    x_cam = (u - cu) / f * z_cam
    y_cam = (v - cv) / f * z_cam
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)
    return X_cam

def compute_pointcloud(depthmap, hfov, camera_position, camera_rotation):
    """Return a 3D point cloud corresponding to valid pixels of the depth map"""
    R_cam2world, t_cam2world = compute_camera_pose_opencv_convention(camera_position, camera_rotation)

    X_cam = compute_pointmap(depthmap=depthmap, hfov=hfov)
    valid_mask = (X_cam[:,:,2] != 0.0)

    X_cam = X_cam.reshape(-1, 3)[valid_mask.flatten()]
    X_world = X_cam @ R_cam2world.T + t_cam2world.reshape(1, 3)
    return X_world

def compute_pointcloud_overlaps_scikit(pointcloud1, pointcloud2, distance_threshold, compute_symmetric=False):
    """
    Compute 'overlapping' metrics based on a distance threshold between two point clouds.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm = 'kd_tree').fit(pointcloud2)
    distances, indices = nbrs.kneighbors(pointcloud1)
    intersection1 = np.count_nonzero(distances.flatten() < distance_threshold)

    data = {"intersection1": intersection1,
            "size1": len(pointcloud1)}
    if compute_symmetric:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm = 'kd_tree').fit(pointcloud1)
        distances, indices = nbrs.kneighbors(pointcloud2)
        intersection2 = np.count_nonzero(distances.flatten() < distance_threshold)
        data["intersection2"] = intersection2
        data["size2"] = len(pointcloud2)

    return data

def _append_camera_parameters(observation, hfov, camera_location, camera_rotation):
    """
    Add camera parameters to the observation dictionnary produced by Habitat-Sim
    In-place modifications.
    """
    R_cam2world, t_cam2world = compute_camera_pose_opencv_convention(camera_location, camera_rotation)
    height, width = observation['depth'].shape
    f, cu, cv = compute_camera_intrinsics(height, width, hfov)
    K = np.asarray([[f, 0, cu],
                    [0, f, cv],
                    [0, 0, 1.0]])
    observation["camera_intrinsics"] = K
    observation["t_cam2world"] = t_cam2world
    observation["R_cam2world"] = R_cam2world

def look_at(eye, center, up, return_cam2world=True):
    """
    Return camera pose looking at a given center point.
    Analogous of gluLookAt function, using OpenCV camera convention.
    """
    z = center - eye
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = -up
    y = y - np.sum(y * z, axis=-1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, z, axis=-1)

    if return_cam2world:
        R = np.stack((x, y, z), axis=-1)
        t = eye
    else:
        # World to camera transformation
        # Transposed matrix
        R = np.stack((x, y, z), axis=-2)
        t = - np.einsum('...ij, ...j', R, eye)
    return R, t

def look_at_for_habitat(eye, center, up, return_cam2world=True):
    R, t = look_at(eye, center, up)
    orientation = quaternion.from_rotation_matrix(R @ R_OPENCV2HABITAT.T)
    return orientation, t

def generate_orientation_noise(pan_range, tilt_range, roll_range):
    return (quaternion.from_rotation_vector(np.random.uniform(*pan_range) * DEG2RAD * habitat_sim.geo.UP)
            * quaternion.from_rotation_vector(np.random.uniform(*tilt_range) * DEG2RAD * habitat_sim.geo.RIGHT)
            * quaternion.from_rotation_vector(np.random.uniform(*roll_range) * DEG2RAD * habitat_sim.geo.FRONT))


class NoNaviguableSpaceError(RuntimeError):
    def __init__(self, *args):
            super().__init__(*args)

class MultiviewHabitatSimGenerator:
    def __init__(self,
                scene,
                navmesh,
                scene_dataset_config_file,
                resolution = (240, 320),
                views_count=2,
                hfov = 60,
                gpu_id = 0,
                size = 10000,
                minimum_covisibility = 0.5,
                transform = None):
        self.scene = scene
        self.navmesh = navmesh
        self.scene_dataset_config_file = scene_dataset_config_file
        self.resolution = resolution
        self.views_count = views_count
        assert(self.views_count >= 1)
        self.hfov = hfov
        self.gpu_id = gpu_id
        self.size = size
        self.transform = transform

        # Noise added to camera orientation
        self.pan_range = (-3, 3)
        self.tilt_range = (-10, 10)
        self.roll_range = (-5, 5)

        # Height range to sample cameras
        self.height_range = (1.2, 1.8)

        # Random steps between the camera views
        self.random_steps_count = 5
        self.random_step_variance = 2.0

        # Minimum fraction of the scene which should be valid (well defined depth)
        self.minimum_valid_fraction = 0.7

        # Distance threshold to see  to select pairs
        self.distance_threshold = 0.05
        # Minimum IoU of a view point cloud with respect to the reference view to be kept.
        self.minimum_covisibility = minimum_covisibility

        # Maximum number of retries.
        self.max_attempts_count = 100

        self.seed = None
        self._lazy_initialization()

    def _lazy_initialization(self):
        # Lazy random seeding and instantiation of the simulator to deal with multiprocessing properly
        if self.seed == None:
            # Re-seed numpy generator
            np.random.seed()
            self.seed = np.random.randint(2**32-1)
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = self.scene
            if self.scene_dataset_config_file is not None and self.scene_dataset_config_file != "":
                    sim_cfg.scene_dataset_config_file = self.scene_dataset_config_file
            sim_cfg.random_seed = self.seed
            sim_cfg.load_semantic_mesh = False
            sim_cfg.gpu_device_id = self.gpu_id

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = self.resolution
            depth_sensor_spec.hfov = self.hfov
            depth_sensor_spec.position = [0.0, 0.0, 0]
            depth_sensor_spec.orientation

            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = self.resolution
            rgb_sensor_spec.hfov = self.hfov
            rgb_sensor_spec.position = [0.0, 0.0, 0]
            agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=[rgb_sensor_spec, depth_sensor_spec])

            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            if self.navmesh is not None and self.navmesh != "":
                # Use pre-computed navmesh when available (usually better than those generated automatically)
                self.sim.pathfinder.load_nav_mesh(self.navmesh)

            if not self.sim.pathfinder.is_loaded:
                # Try to compute a navmesh
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings, True)

            # Ensure that the navmesh is not empty
            if not self.sim.pathfinder.is_loaded:
                raise NoNaviguableSpaceError(f"No naviguable location (scene: {self.scene} -- navmesh: {self.navmesh})")

            self.agent = self.sim.initialize_agent(agent_id=0)

    def close(self):
        self.sim.close()

    def __del__(self):
        self.sim.close()

    def __len__(self):
        return self.size

    def sample_random_viewpoint(self):
        """ Sample a random viewpoint using the navmesh """
        nav_point = self.sim.pathfinder.get_random_navigable_point()

        # Sample a random viewpoint height
        viewpoint_height = np.random.uniform(*self.height_range)
        viewpoint_position = nav_point + viewpoint_height * habitat_sim.geo.UP
        viewpoint_orientation = quaternion.from_rotation_vector(np.random.uniform(0, 2 * np.pi) * habitat_sim.geo.UP) * generate_orientation_noise(self.pan_range, self.tilt_range, self.roll_range)
        return viewpoint_position, viewpoint_orientation, nav_point

    def sample_other_random_viewpoint(self, observed_point, nav_point):
        """ Sample a random viewpoint close to an existing one, using the navmesh and a reference observed point."""
        other_nav_point = nav_point

        walk_directions = self.random_step_variance * np.asarray([1,0,1])
        for i in range(self.random_steps_count):
            temp = self.sim.pathfinder.snap_point(other_nav_point + walk_directions * np.random.normal(size=3))
            # Snapping may return nan when it fails
            if not np.isnan(temp[0]):
                    other_nav_point = temp

        other_viewpoint_height = np.random.uniform(*self.height_range)
        other_viewpoint_position = other_nav_point + other_viewpoint_height * habitat_sim.geo.UP

        # Set viewing direction towards the central point
        rotation, position = look_at_for_habitat(eye=other_viewpoint_position, center=observed_point, up=habitat_sim.geo.UP, return_cam2world=True)
        rotation = rotation * generate_orientation_noise(self.pan_range, self.tilt_range, self.roll_range)
        return position, rotation, other_nav_point

    def is_other_pointcloud_overlapping(self, ref_pointcloud, other_pointcloud):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        pixels_count = self.resolution[0] * self.resolution[1]
        valid_fraction = len(other_pointcloud) / pixels_count
        assert valid_fraction <= 1.0 and valid_fraction >= 0.0
        overlap = compute_pointcloud_overlaps_scikit(ref_pointcloud, other_pointcloud, self.distance_threshold, compute_symmetric=True)
        covisibility = min(overlap["intersection1"] / pixels_count, overlap["intersection2"] / pixels_count)
        is_valid = (valid_fraction >= self.minimum_valid_fraction) and (covisibility >= self.minimum_covisibility)
        return is_valid, valid_fraction, covisibility

    def is_other_viewpoint_overlapping(self, ref_pointcloud, observation, position, rotation):
        """ Check if a viewpoint is valid and overlaps significantly with a reference one. """
        # Observation
        other_pointcloud = compute_pointcloud(observation['depth'], self.hfov, position, rotation)
        return self.is_other_pointcloud_overlapping(ref_pointcloud, other_pointcloud)

    def render_viewpoint(self, viewpoint_position, viewpoint_orientation):
        agent_state = habitat_sim.AgentState()
        agent_state.position = viewpoint_position
        agent_state.rotation = viewpoint_orientation
        self.agent.set_state(agent_state)
        viewpoint_observations = self.sim.get_sensor_observations(agent_ids=0)
        _append_camera_parameters(viewpoint_observations, self.hfov, viewpoint_position, viewpoint_orientation)
        return viewpoint_observations

    def __getitem__(self, useless_idx):
        ref_position, ref_orientation, nav_point = self.sample_random_viewpoint()
        ref_observations = self.render_viewpoint(ref_position, ref_orientation)
        # Extract point cloud
        ref_pointcloud = compute_pointcloud(depthmap=ref_observations['depth'], hfov=self.hfov,
                                        camera_position=ref_position, camera_rotation=ref_orientation)

        pixels_count = self.resolution[0] * self.resolution[1]
        ref_valid_fraction = len(ref_pointcloud) / pixels_count
        assert ref_valid_fraction <= 1.0 and ref_valid_fraction >= 0.0
        if ref_valid_fraction < self.minimum_valid_fraction:
                # This should produce a recursion error at some point when something is very wrong.
                return self[0]
        # Pick an reference observed point in the point cloud
        observed_point = np.mean(ref_pointcloud, axis=0)

        # Add the first image as reference
        viewpoints_observations = [ref_observations]
        viewpoints_covisibility = [ref_valid_fraction]
        viewpoints_positions = [ref_position]
        viewpoints_orientations = [quaternion.as_float_array(ref_orientation)]
        viewpoints_clouds = [ref_pointcloud]
        viewpoints_valid_fractions = [ref_valid_fraction]

        for _ in range(self.views_count - 1):
            # Generate an other viewpoint using some dummy random walk
            successful_sampling = False
            for sampling_attempt in range(self.max_attempts_count):
                position, rotation, _ = self.sample_other_random_viewpoint(observed_point, nav_point)
                # Observation
                other_viewpoint_observations = self.render_viewpoint(position, rotation)
                other_pointcloud = compute_pointcloud(other_viewpoint_observations['depth'], self.hfov, position, rotation)

                is_valid, valid_fraction, covisibility = self.is_other_pointcloud_overlapping(ref_pointcloud, other_pointcloud)
                if is_valid:
                        successful_sampling = True
                        break
            if not successful_sampling:
                print("WARNING: Maximum number of attempts reached.")
                # Dirty hack, try using a novel original viewpoint
                return self[0]
            viewpoints_observations.append(other_viewpoint_observations)
            viewpoints_covisibility.append(covisibility)
            viewpoints_positions.append(position)
            viewpoints_orientations.append(quaternion.as_float_array(rotation)) # WXYZ convention for the quaternion encoding.
            viewpoints_clouds.append(other_pointcloud)
            viewpoints_valid_fractions.append(valid_fraction)

        # Estimate relations between all pairs of images
        pairwise_visibility_ratios = np.ones((len(viewpoints_observations), len(viewpoints_observations)))
        for i in range(len(viewpoints_observations)):
            pairwise_visibility_ratios[i,i] = viewpoints_valid_fractions[i]
            for j in range(i+1, len(viewpoints_observations)):
                overlap = compute_pointcloud_overlaps_scikit(viewpoints_clouds[i], viewpoints_clouds[j], self.distance_threshold, compute_symmetric=True)
                pairwise_visibility_ratios[i,j] = overlap['intersection1'] / pixels_count
                pairwise_visibility_ratios[j,i] = overlap['intersection2'] / pixels_count

        # IoU is relative to the image 0
        data = {"observations": viewpoints_observations,
                "positions": np.asarray(viewpoints_positions),
                "orientations": np.asarray(viewpoints_orientations),
                "covisibility_ratios": np.asarray(viewpoints_covisibility),
                "valid_fractions": np.asarray(viewpoints_valid_fractions, dtype=float),
                "pairwise_visibility_ratios": np.asarray(pairwise_visibility_ratios, dtype=float),
                }

        if self.transform is not None:
            data = self.transform(data)
        return  data

    def generate_random_spiral_trajectory(self, images_count = 100, max_radius=0.5, half_turns=5, use_constant_orientation=False):
        """
        Return a list of images corresponding to a spiral trajectory from a random starting point.
        Useful to generate nice visualisations.
        Use an even number of half turns to get a nice "C1-continuous" loop effect 
        """
        ref_position, ref_orientation, navpoint = self.sample_random_viewpoint()
        ref_observations = self.render_viewpoint(ref_position, ref_orientation)
        ref_pointcloud = compute_pointcloud(depthmap=ref_observations['depth'], hfov=self.hfov,
                                                        camera_position=ref_position, camera_rotation=ref_orientation)
        pixels_count = self.resolution[0] * self.resolution[1]
        if len(ref_pointcloud) / pixels_count < self.minimum_valid_fraction:
            # Dirty hack: ensure that the valid part of the image is significant
            return self.generate_random_spiral_trajectory(images_count, max_radius, half_turns, use_constant_orientation)

        # Pick an observed point in the point cloud
        observed_point = np.mean(ref_pointcloud, axis=0)
        ref_R, ref_t = compute_camera_pose_opencv_convention(ref_position, ref_orientation)

        images = []
        is_valid = []
        # Spiral trajectory, use_constant orientation
        for i, alpha in enumerate(np.linspace(0, 1, images_count)):
            r = max_radius * np.abs(np.sin(alpha * np.pi)) # Increase then decrease the radius
            theta = alpha * half_turns * np.pi 
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0.0
            position = ref_position + (ref_R @ np.asarray([x, y, z]).reshape(3,1)).flatten()
            if use_constant_orientation:
                orientation = ref_orientation
            else:
                # trajectory looking at a mean point in front of the ref observation
                orientation, position = look_at_for_habitat(eye=position, center=observed_point, up=habitat_sim.geo.UP)
            observations = self.render_viewpoint(position, orientation)
            images.append(observations['color'][...,:3])
            _is_valid, valid_fraction, iou = self.is_other_viewpoint_overlapping(ref_pointcloud, observations, position, orientation)
            is_valid.append(_is_valid)
        return images, np.all(is_valid)