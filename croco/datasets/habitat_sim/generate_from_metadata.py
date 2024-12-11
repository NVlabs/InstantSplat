# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""
Script to generate image pairs for a given scene reproducing poses provided in a metadata file.
"""
import os
from datasets.habitat_sim.multiview_habitat_sim_generator import MultiviewHabitatSimGenerator
from datasets.habitat_sim.paths import SCENES_DATASET
import argparse
import quaternion
import PIL.Image
import cv2
import json
from tqdm import tqdm

def generate_multiview_images_from_metadata(metadata_filename,
                                            output_dir,
                                            overload_params = dict(),
                                            scene_datasets_paths=None,
                                            exist_ok=False):   
    """
    Generate images from a metadata file for reproducibility purposes.
    """
    # Reorder paths by decreasing label length, to avoid collisions when testing if a string by such label
    if scene_datasets_paths is not None:
        scene_datasets_paths = dict(sorted(scene_datasets_paths.items(), key= lambda x: len(x[0]), reverse=True))

    with open(metadata_filename, 'r') as f:
        input_metadata = json.load(f)
    metadata = dict()
    for key, value in input_metadata.items():
        # Optionally replace some paths
        if key in ("scene_dataset_config_file", "scene", "navmesh") and value != "":
            if scene_datasets_paths is not None:
                for dataset_label, dataset_path in scene_datasets_paths.items():
                    if value.startswith(dataset_label):
                        value = os.path.normpath(os.path.join(dataset_path, os.path.relpath(value, dataset_label)))
                        break
        metadata[key] = value

    # Overload some parameters
    for key, value in overload_params.items():
        metadata[key] = value

    generation_entries = dict([(key, value) for key, value in metadata.items() if not (key in ('multiviews', 'output_dir', 'generate_depth'))])
    generate_depth = metadata["generate_depth"]

    os.makedirs(output_dir, exist_ok=exist_ok)
 
    generator = MultiviewHabitatSimGenerator(**generation_entries)

    # Generate views
    for idx_label, data in tqdm(metadata['multiviews'].items()):
        positions = data["positions"]
        orientations = data["orientations"]
        n = len(positions)
        for oidx in range(n):
            observation = generator.render_viewpoint(positions[oidx], quaternion.from_float_array(orientations[oidx]))
            observation_label = f"{oidx + 1}" # Leonid is indexing starting from 1
            # Color image saved using PIL
            img = PIL.Image.fromarray(observation['color'][:,:,:3])
            filename = os.path.join(output_dir, f"{idx_label}_{observation_label}.jpeg")
            img.save(filename)
            if generate_depth:
                # Depth image as EXR file
                filename = os.path.join(output_dir, f"{idx_label}_{observation_label}_depth.exr")
                cv2.imwrite(filename, observation['depth'], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                # Camera parameters
                camera_params = dict([(key, observation[key].tolist()) for key in ("camera_intrinsics", "R_cam2world", "t_cam2world")])
                filename = os.path.join(output_dir, f"{idx_label}_{observation_label}_camera_params.json")
                with open(filename, "w") as f:
                    json.dump(camera_params, f)
                # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    generator.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_filename", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    generate_multiview_images_from_metadata(metadata_filename=args.metadata_filename,
                             output_dir=args.output_dir,
                             scene_datasets_paths=SCENES_DATASET,
                             overload_params=dict(),
                             exist_ok=True)

 