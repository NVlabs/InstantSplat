# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
from tqdm import tqdm
import argparse
import PIL.Image
import numpy as np
import json
from datasets.habitat_sim.multiview_habitat_sim_generator import MultiviewHabitatSimGenerator, NoNaviguableSpaceError
from datasets.habitat_sim.paths import list_scenes_available
import cv2
import quaternion
import shutil

def generate_multiview_images_for_scene(scene_dataset_config_file,
                                        scene,
                                        navmesh,
                                        output_dir, 
                                        views_count,
                                        size, 
                                        exist_ok=False, 
                                        generate_depth=False,
                                        **kwargs):
    """
    Generate tuples of overlapping views for a given scene.
    generate_depth: generate depth images and camera parameters.
    """
    if os.path.exists(output_dir) and not exist_ok:
        print(f"Scene {scene}: data already generated. Ignoring generation.")
        return
    try:
        print(f"Scene {scene}: {size} multiview acquisitions to generate...")
        os.makedirs(output_dir, exist_ok=exist_ok)

        metadata_filename = os.path.join(output_dir, "metadata.json")

        metadata_template = dict(scene_dataset_config_file=scene_dataset_config_file,
            scene=scene, 
            navmesh=navmesh,
            views_count=views_count,
            size=size,
            generate_depth=generate_depth,
            **kwargs)
        metadata_template["multiviews"] = dict()

        if os.path.exists(metadata_filename):
            print("Metadata file already exists:", metadata_filename)
            print("Loading already generated metadata file...")
            with open(metadata_filename, "r") as f:
                metadata = json.load(f)

            for key in metadata_template.keys():
                if key != "multiviews":
                    assert metadata_template[key] == metadata[key], f"existing file is inconsistent with the input parameters:\nKey: {key}\nmetadata: {metadata[key]}\ntemplate: {metadata_template[key]}."
        else:
            print("No temporary file found. Starting generation from scratch...")
            metadata = metadata_template

        starting_id = len(metadata["multiviews"])
        print(f"Starting generation from index {starting_id}/{size}...")
        if starting_id >= size:
            print("Generation already done.")
            return

        generator = MultiviewHabitatSimGenerator(scene_dataset_config_file=scene_dataset_config_file,
                                                scene=scene,
                                                navmesh=navmesh,
                                                views_count = views_count,
                                                size = size,
                                                **kwargs)

        for idx in tqdm(range(starting_id, size)):
            # Generate / re-generate the observations
            try:
                data = generator[idx]
                observations = data["observations"]
                positions = data["positions"]
                orientations = data["orientations"]

                idx_label = f"{idx:08}"
                for oidx, observation in enumerate(observations):
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
                metadata["multiviews"][idx_label] = {"positions": positions.tolist(),
                                                    "orientations": orientations.tolist(),
                                                    "covisibility_ratios": data["covisibility_ratios"].tolist(),
                                                    "valid_fractions": data["valid_fractions"].tolist(),
                                                    "pairwise_visibility_ratios": data["pairwise_visibility_ratios"].tolist()}
            except RecursionError:
                print("Recursion error: unable to sample observations for this scene. We will stop there.")
                break

            # Regularly save a temporary metadata file, in case we need to restart the generation
            if idx % 10 == 0:
                with open(metadata_filename, "w") as f:
                    json.dump(metadata, f)

        # Save metadata
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f)

        generator.close()
    except NoNaviguableSpaceError:
        pass

def create_commandline(scene_data, generate_depth, exist_ok=False):
    """
    Create a commandline string to generate a scene.
    """
    def my_formatting(val):
        if val is None or val == "":
            return '""'
        else:
            return val
    commandline = f"""python {__file__} --scene {my_formatting(scene_data.scene)} 
    --scene_dataset_config_file {my_formatting(scene_data.scene_dataset_config_file)} 
    --navmesh {my_formatting(scene_data.navmesh)} 
    --output_dir {my_formatting(scene_data.output_dir)} 
    --generate_depth {int(generate_depth)} 
    --exist_ok {int(exist_ok)}
    """
    commandline = " ".join(commandline.split())
    return commandline

if __name__ == "__main__":
    os.umask(2)

    parser = argparse.ArgumentParser(description="""Example of use -- listing commands to generate data for scenes available:
    > python datasets/habitat_sim/generate_multiview_habitat_images.py --list_commands
    """)
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--list_commands", action='store_true', help="list commandlines to run if true")
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--scene_dataset_config_file", type=str, default="")
    parser.add_argument("--navmesh", type=str, default="")
    
    parser.add_argument("--generate_depth", type=int, default=1)
    parser.add_argument("--exist_ok", type=int, default=0)

    kwargs = dict(resolution=(256,256), hfov=60, views_count = 2, size=1000)

    args = parser.parse_args()
    generate_depth=bool(args.generate_depth)
    exist_ok = bool(args.exist_ok)

    if args.list_commands:
        # Listing scenes available...
        scenes_data = list_scenes_available(base_output_dir=args.output_dir)
        
        for scene_data in scenes_data:
            print(create_commandline(scene_data, generate_depth=generate_depth, exist_ok=exist_ok))
    else:
        if args.scene == "" or args.output_dir == "":
            print("Missing scene or output dir argument!")
            print(parser.format_help())
        else:
            generate_multiview_images_for_scene(scene=args.scene,
                                                scene_dataset_config_file = args.scene_dataset_config_file,
                                                navmesh = args.navmesh,
                                                output_dir = args.output_dir,
                                                exist_ok=exist_ok,
                                                generate_depth=generate_depth,
                                                **kwargs)