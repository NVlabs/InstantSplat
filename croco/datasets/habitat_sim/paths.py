# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""
Paths to Habitat-Sim scenes
"""

import os
import json
import collections
from tqdm import tqdm


# Hardcoded path to the different scene datasets
SCENES_DATASET = {
    "hm3d": "./data/habitat-sim-data/scene_datasets/hm3d/",
    "gibson": "./data/habitat-sim-data/scene_datasets/gibson/",
    "habitat-test-scenes": "./data/habitat-sim/scene_datasets/habitat-test-scenes/",
    "replica_cad_baked_lighting": "./data/habitat-sim/scene_datasets/replica_cad_baked_lighting/",
    "replica_cad": "./data/habitat-sim/scene_datasets/replica_cad/",
    "replica": "./data/habitat-sim/scene_datasets/ReplicaDataset/",
    "scannet": "./data/habitat-sim/scene_datasets/scannet/"
}

SceneData = collections.namedtuple("SceneData", ["scene_dataset_config_file", "scene", "navmesh", "output_dir"])

def list_replicacad_scenes(base_output_dir, base_path=SCENES_DATASET["replica_cad"]):
    scene_dataset_config_file = os.path.join(base_path, "replicaCAD.scene_dataset_config.json")
    scenes = [f"apt_{i}" for i in range(6)] + ["empty_stage"]
    navmeshes = [f"navmeshes/apt_{i}_static_furniture.navmesh" for i in range(6)] + ["empty_stage.navmesh"]
    scenes_data = []
    for idx in range(len(scenes)):
        output_dir = os.path.join(base_output_dir, "ReplicaCAD", scenes[idx])
        # Add scene
        data = SceneData(scene_dataset_config_file=scene_dataset_config_file,
                    scene = scenes[idx] + ".scene_instance.json",
                    navmesh = os.path.join(base_path, navmeshes[idx]),
                    output_dir = output_dir)
        scenes_data.append(data)
    return scenes_data

def list_replica_cad_baked_lighting_scenes(base_output_dir, base_path=SCENES_DATASET["replica_cad_baked_lighting"]):
    scene_dataset_config_file = os.path.join(base_path, "replicaCAD_baked.scene_dataset_config.json")
    scenes = sum([[f"Baked_sc{i}_staging_{j:02}" for i in range(5)] for j in range(21)], [])
    navmeshes = ""#[f"navmeshes/apt_{i}_static_furniture.navmesh" for i in range(6)] + ["empty_stage.navmesh"]
    scenes_data = []
    for idx in range(len(scenes)):
        output_dir = os.path.join(base_output_dir, "replica_cad_baked_lighting", scenes[idx])
        data = SceneData(scene_dataset_config_file=scene_dataset_config_file,
                    scene = scenes[idx],
                    navmesh = "",
                    output_dir = output_dir)
        scenes_data.append(data)
    return scenes_data    

def list_replica_scenes(base_output_dir, base_path):
    scenes_data = []
    for scene_id in os.listdir(base_path):
        scene = os.path.join(base_path, scene_id, "mesh.ply")
        navmesh = os.path.join(base_path, scene_id, "habitat/mesh_preseg_semantic.navmesh") # Not sure if I should use it
        scene_dataset_config_file = ""
        output_dir = os.path.join(base_output_dir, scene_id)
        # Add scene only if it does not exist already, or if exist_ok
        data = SceneData(scene_dataset_config_file = scene_dataset_config_file,
                    scene = scene,
                    navmesh = navmesh,
                    output_dir = output_dir)
        scenes_data.append(data)
    return scenes_data


def list_scenes(base_output_dir, base_path):
    """
    Generic method iterating through a base_path folder to find scenes.
    """
    scenes_data = []
    for root, dirs, files in os.walk(base_path, followlinks=True):
        folder_scenes_data = []
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == ".glb":
                scene = os.path.join(root, name + ".glb")
                navmesh = os.path.join(root, name + ".navmesh")
                if not os.path.exists(navmesh):
                    navmesh = ""
                relpath = os.path.relpath(root, base_path)
                output_dir = os.path.abspath(os.path.join(base_output_dir, relpath, name))
                data = SceneData(scene_dataset_config_file="",
                    scene = scene,
                    navmesh = navmesh,
                    output_dir = output_dir)
                folder_scenes_data.append(data)

        # Specific check for HM3D:
        # When two meshesxxxx.basis.glb and xxxx.glb are present, use the 'basis' version.
        basis_scenes = [data.scene[:-len(".basis.glb")] for data in folder_scenes_data if data.scene.endswith(".basis.glb")]
        if len(basis_scenes) != 0:
            folder_scenes_data = [data for data in folder_scenes_data if not (data.scene[:-len(".glb")] in basis_scenes)]

        scenes_data.extend(folder_scenes_data)
    return scenes_data

def list_scenes_available(base_output_dir, scenes_dataset_paths=SCENES_DATASET):
    scenes_data = []

    # HM3D
    for split in ("minival", "train", "val", "examples"):
        scenes_data += list_scenes(base_output_dir=os.path.join(base_output_dir, f"hm3d/{split}/"),
                                    base_path=f"{scenes_dataset_paths['hm3d']}/{split}")

    # Gibson
    scenes_data += list_scenes(base_output_dir=os.path.join(base_output_dir, "gibson"),
                                base_path=scenes_dataset_paths["gibson"])

    # Habitat test scenes (just a few)
    scenes_data += list_scenes(base_output_dir=os.path.join(base_output_dir, "habitat-test-scenes"),
                                base_path=scenes_dataset_paths["habitat-test-scenes"])

    # ReplicaCAD (baked lightning)
    scenes_data += list_replica_cad_baked_lighting_scenes(base_output_dir=base_output_dir)

    # ScanNet
    scenes_data += list_scenes(base_output_dir=os.path.join(base_output_dir, "scannet"), 
                            base_path=scenes_dataset_paths["scannet"])
    
    # Replica
    list_replica_scenes(base_output_dir=os.path.join(base_output_dir, "replica"),
                        base_path=scenes_dataset_paths["replica"])
    return scenes_data    
