# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""
Utility script to pack metadata files of the dataset in order to be able to re-generate it elsewhere.
"""
import os
import glob
from tqdm import tqdm
import shutil
import json
from datasets.habitat_sim.paths import *
import argparse
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dirname = args.input_dir
    output_dirname = args.output_dir

    input_metadata_filenames = glob.iglob(f"{input_dirname}/**/metadata.json", recursive=True)

    images_count = collections.defaultdict(lambda : 0)
    
    os.makedirs(output_dirname)
    for input_filename in tqdm(input_metadata_filenames):
        # Ignore empty files
        with open(input_filename, "r") as f:
            original_metadata = json.load(f)
            if "multiviews" not in original_metadata or len(original_metadata["multiviews"]) == 0:
                print("No views in", input_filename)
                continue

        relpath = os.path.relpath(input_filename, input_dirname)
        print(relpath)

        # Copy metadata, while replacing scene paths by generic keys depending on the dataset, for portability.
        # Data paths are sorted by decreasing length to avoid potential bugs due to paths starting by the same string pattern.
        scenes_dataset_paths = dict(sorted(SCENES_DATASET.items(), key=lambda x: len(x[1]), reverse=True))
        metadata = dict()
        for key, value in original_metadata.items():
            if key in ("scene_dataset_config_file", "scene", "navmesh") and value != "":
                known_path = False
                for dataset, dataset_path in scenes_dataset_paths.items():
                    if value.startswith(dataset_path):
                        value = os.path.join(dataset, os.path.relpath(value, dataset_path))
                        known_path = True
                        break
                if not known_path:
                    raise KeyError("Unknown path:" + value)
            metadata[key] = value

        # Compile some general statistics while packing data
        scene_split = metadata["scene"].split("/")
        upper_level = "/".join(scene_split[:2]) if scene_split[0] == "hm3d" else scene_split[0]
        images_count[upper_level] += len(metadata["multiviews"])
        
        output_filename = os.path.join(output_dirname, relpath)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(metadata, f)

    # Print statistics
    print("Images count:")
    for upper_level, count in images_count.items():
        print(f"- {upper_level}: {count}")