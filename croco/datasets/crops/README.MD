## Generation of crops from the real datasets

The instructions below allow to generate the crops used for pre-training CroCo v2 from the following real-world datasets: ARKitScenes, MegaDepth, 3DStreetView and IndoorVL.

### Download the metadata of the crops to generate 

First, download the metadata and put them in `./data/`:
```
mkdir -p data
cd data/
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/data/crop_metadata.zip
unzip crop_metadata.zip
rm crop_metadata.zip
cd ..
```

### Prepare the original datasets 

Second, download the original datasets in `./data/original_datasets/`.
```
mkdir -p data/original_datasets
```

##### ARKitScenes

Download the `raw` dataset from https://github.com/apple/ARKitScenes/blob/main/DATA.md and put it in `./data/original_datasets/ARKitScenes/`.
The resulting file structure should be like:
```
./data/original_datasets/ARKitScenes/
└───Training
    └───40753679
     │  │   ultrawide
     │  │   ...
     └───40753686
     │   
      ...
```

##### MegaDepth

Download `MegaDepth v1 Dataset` from https://www.cs.cornell.edu/projects/megadepth/ and put it in `./data/original_datasets/MegaDepth/`.
The resulting file structure should be like:

```
./data/original_datasets/MegaDepth/
└───0000
│   └───images
│    │      │   1000557903_87fa96b8a4_o.jpg
│    │      └ ...
│    └─── ...
└───0001
│   │   
│   └ ...
└─── ...
```

##### 3DStreetView

Download `3D_Street_View` dataset from https://github.com/amir32002/3D_Street_View and put it in `./data/original_datasets/3DStreetView/`.
The resulting file structure should be like:

``` 
./data/original_datasets/3DStreetView/
└───dataset_aligned
│   └───0002
│    │      │   0000002_0000001_0000002_0000001.jpg
│    │      └ ...
│    └─── ...
└───dataset_unaligned
│   └───0003
│    │      │   0000003_0000001_0000002_0000001.jpg
│    │      └ ...
│    └─── ...
```

##### IndoorVL

Download the `IndoorVL` datasets using [Kapture](https://github.com/naver/kapture).

```
pip install kapture
mkdir -p ./data/original_datasets/IndoorVL
cd ./data/original_datasets/IndoorVL
kapture_download_dataset.py update
kapture_download_dataset.py install  "HyundaiDepartmentStore_*"
kapture_download_dataset.py install  "GangnamStation_*"
cd -
```

### Extract the crops

Now, extract the crops for each of the dataset:
```
for dataset in ARKitScenes MegaDepth 3DStreetView IndoorVL; 
do 
  python3 datasets/crops/extract_crops_from_images.py --crops ./data/crop_metadata/${dataset}/crops_release.txt --root-dir ./data/original_datasets/${dataset}/ --output-dir ./data/${dataset}_crops/ --imsize 256 --nthread 8 --max-subdir-levels 5 --ideal-number-pairs-in-dir 500;
done
```

##### Note for IndoorVL

Due to some legal issues, we can only release 144,228 pairs out of the 1,593,689 pairs used in the paper.
To account for it in terms of number of pre-training iterations, the pre-training command in this repository uses 125 training epochs including 12 warm-up epochs and learning rate cosine schedule of 250, instead of 100, 10 and 200 respectively.
The impact on the performance is negligible.
