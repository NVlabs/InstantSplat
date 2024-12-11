## Generation of synthetic image pairs using Habitat-Sim

These instructions allow to generate pre-training pairs from the Habitat simulator.
As we did not save metadata of the pairs used in the original paper, they are not strictly the same, but these data use the same setting and are equivalent.

### Download Habitat-Sim scenes
Download Habitat-Sim scenes:
- Download links can be found here: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
- We used scenes from the HM3D, habitat-test-scenes, Replica, ReplicaCad and ScanNet datasets.
- Please put the scenes under `./data/habitat-sim-data/scene_datasets/` following the structure below, or update manually paths in `paths.py`.
```
./data/
└──habitat-sim-data/
   └──scene_datasets/
      ├──hm3d/
      ├──gibson/
      ├──habitat-test-scenes/
      ├──replica_cad_baked_lighting/
      ├──replica_cad/
      ├──ReplicaDataset/
      └──scannet/
```

### Image pairs generation
We provide metadata to generate reproducible images pairs for pretraining and validation.
Experiments described in the paper used similar data, but whose generation was not reproducible at the time.

Specifications:
- 256x256 resolution images, with 60 degrees field of view .
- Up to 1000 image pairs per scene.
- Number of scenes considered/number of images pairs per dataset:
  - Scannet: 1097 scenes / 985 209 pairs
  - HM3D:
    - hm3d/train: 800 / 800k pairs
    - hm3d/val: 100 scenes / 100k pairs
    - hm3d/minival: 10 scenes / 10k pairs
  - habitat-test-scenes: 3 scenes / 3k pairs
  - replica_cad_baked_lighting: 13 scenes / 13k pairs

- Scenes from hm3d/val and hm3d/minival pairs were not used for the pre-training but kept for validation purposes.

Download metadata and extract it:
```bash
mkdir -p data/habitat_release_metadata/
cd data/habitat_release_metadata/
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/data/habitat_release_metadata/multiview_habitat_metadata.tar.gz
tar -xvf multiview_habitat_metadata.tar.gz
cd ../..
# Location of the metadata
METADATA_DIR="./data/habitat_release_metadata/multiview_habitat_metadata"
```

Generate image pairs from metadata:
- The following command will print a list of commandlines to generate image pairs for each scene:
```bash
# Target output directory
PAIRS_DATASET_DIR="./data/habitat_release/"
python datasets/habitat_sim/generate_from_metadata_files.py --input_dir=$METADATA_DIR --output_dir=$PAIRS_DATASET_DIR
```
- One can launch multiple of such commands in parallel e.g. using GNU Parallel:
```bash
python datasets/habitat_sim/generate_from_metadata_files.py --input_dir=$METADATA_DIR --output_dir=$PAIRS_DATASET_DIR | parallel -j 16
```

## Metadata generation

Image pairs were randomly sampled using the following commands, whose outputs contain randomness and are thus not exactly reproducible:
```bash
# Print commandlines to generate image pairs from the different scenes available.
PAIRS_DATASET_DIR=MY_CUSTOM_PATH
python datasets/habitat_sim/generate_multiview_images.py --list_commands --output_dir=$PAIRS_DATASET_DIR

# Once a dataset is generated, pack metadata files for reproducibility.
METADATA_DIR=MY_CUSTON_PATH
python datasets/habitat_sim/pack_metadata_files.py $PAIRS_DATASET_DIR  $METADATA_DIR
```
