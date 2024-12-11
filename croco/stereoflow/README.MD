## CroCo-Stereo and CroCo-Flow

This README explains how to use CroCo-Stereo and CroCo-Flow as well as how they were trained.
All commands should be launched from the root directory.

### Simple inference example

We provide a simple inference exemple for CroCo-Stereo and CroCo-Flow in the Totebook `croco-stereo-flow-demo.ipynb`.
Before running it, please download the trained models with:
```
bash stereoflow/download_model.sh crocostereo.pth
bash stereoflow/download_model.sh crocoflow.pth
```

### Prepare data for training or evaluation

Put the datasets used for training/evaluation in `./data/stereoflow` (or update the paths at the top of `stereoflow/datasets_stereo.py` and `stereoflow/datasets_flow.py`).
Please find below on the file structure should look for each dataset:
<details>
<summary>FlyingChairs</summary>

```
./data/stereoflow/FlyingChairs/
└───chairs_split.txt
└───data/
    └─── ...
```
</details>

<details>
<summary>MPI-Sintel</summary>

```
./data/stereoflow/MPI-Sintel/
└───training/
│   └───clean/
│   └───final/
│   └───flow/
└───test/
    └───clean/
    └───final/
```
</details>

<details>
<summary>SceneFlow (including FlyingThings)</summary>

```
./data/stereoflow/SceneFlow/
└───Driving/
│   └───disparity/
│   └───frames_cleanpass/
│   └───frames_finalpass/
└───FlyingThings/
│   └───disparity/
│   └───frames_cleanpass/
│   └───frames_finalpass/
│   └───optical_flow/
└───Monkaa/
    └───disparity/
    └───frames_cleanpass/
    └───frames_finalpass/
```
</details>

<details>
<summary>TartanAir</summary>

```
./data/stereoflow/TartanAir/
└───abandonedfactory/
│   └───.../
└───abandonedfactory_night/
│   └───.../
└───.../
```
</details>

<details>
<summary>Booster</summary>

```
./data/stereoflow/booster_gt/
└───train/
    └───balanced/
        └───Bathroom/
        └───Bedroom/
        └───...
```
</details>

<details>
<summary>CREStereo</summary>

```
./data/stereoflow/crenet_stereo_trainset/
└───stereo_trainset/
    └───crestereo/
        └───hole/
        └───reflective/
        └───shapenet/
        └───tree/
```
</details>

<details>
<summary>ETH3D Two-view Low-res</summary>

```
./data/stereoflow/eth3d_lowres/
└───test/
│   └───lakeside_1l/
│   └───...
└───train/
│   └───delivery_area_1l/
│   └───...
└───train_gt/
    └───delivery_area_1l/
    └───...
```
</details>

<details>
<summary>KITTI 2012</summary>

```
./data/stereoflow/kitti-stereo-2012/
└───testing/
│   └───colored_0/
│   └───colored_1/
└───training/
    └───colored_0/
    └───colored_1/
    └───disp_occ/
    └───flow_occ/
```
</details>

<details>
<summary>KITTI 2015</summary>

```
./data/stereoflow/kitti-stereo-2015/
└───testing/
│   └───image_2/
│   └───image_3/
└───training/
    └───image_2/
    └───image_3/
    └───disp_occ_0/
    └───flow_occ/
```
</details>

<details>
<summary>Middlebury</summary>

```
./data/stereoflow/middlebury
└───2005/
│   └───train/
│       └───Art/
│       └───...
└───2006/
│   └───Aloe/
│   └───Baby1/
│   └───...
└───2014/
│   └───Adirondack-imperfect/
│   └───Adirondack-perfect/
│   └───...
└───2021/
│   └───data/
│       └───artroom1/
│       └───artroom2/
│       └───...
└───MiddEval3_F/
    └───test/
    │   └───Australia/
    │   └───...
    └───train/
        └───Adirondack/
        └───...
```
</details>

<details>
<summary>Spring</summary>

```
./data/stereoflow/spring/
└───test/
│   └───0003/
│   └───...
└───train/
    └───0001/
    └───...
```
</details>


### CroCo-Stereo

##### Main model 

The main training of CroCo-Stereo was performed on a series of datasets, and it was used as it for Middlebury v3 benchmark.

```
# Download the model
bash stereoflow/download_model.sh crocostereo.pth
# Middlebury v3 submission
python stereoflow/test.py --model stereoflow_models/crocostereo.pth --dataset "MdEval3('all_full')" --save submission --tile_overlap 0.9
# Training command that was used, using checkpoint-last.pth
python -u stereoflow/train.py stereo --criterion "LaplacianLossBounded2()" --dataset "CREStereo('train')+SceneFlow('train_allpass')+30*ETH3DLowRes('train')+50*Md05('train')+50*Md06('train')+50*Md14('train')+50*Md21('train')+50*MdEval3('train_full')+Booster('train_balanced')" --val_dataset "SceneFlow('test1of100_finalpass')+SceneFlow('test1of100_cleanpass')+ETH3DLowRes('subval')+Md05('subval')+Md06('subval')+Md14('subval')+Md21('subval')+MdEval3('subval_full')+Booster('subval_balanced')" --lr 3e-5 --batch_size 6 --epochs 32 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --output_dir xps/crocostereo/main/
# or it can be launched on multiple gpus (while maintaining the effective batch size), e.g. on 3 gpus:
torchrun --nproc_per_node 3 stereoflow/train.py stereo --criterion "LaplacianLossBounded2()" --dataset "CREStereo('train')+SceneFlow('train_allpass')+30*ETH3DLowRes('train')+50*Md05('train')+50*Md06('train')+50*Md14('train')+50*Md21('train')+50*MdEval3('train_full')+Booster('train_balanced')" --val_dataset "SceneFlow('test1of100_finalpass')+SceneFlow('test1of100_cleanpass')+ETH3DLowRes('subval')+Md05('subval')+Md06('subval')+Md14('subval')+Md21('subval')+MdEval3('subval_full')+Booster('subval_balanced')" --lr 3e-5 --batch_size 2 --epochs 32 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --output_dir xps/crocostereo/main/
```

For evaluation of validation set, we also provide the model trained on the `subtrain` subset of the training sets.

```
# Download the model
bash stereoflow/download_model.sh crocostereo_subtrain.pth
# Evaluation on validation sets 
python stereoflow/test.py --model stereoflow_models/crocostereo_subtrain.pth --dataset "MdEval3('subval_full')+ETH3DLowRes('subval')+SceneFlow('test_finalpass')+SceneFlow('test_cleanpass')" --save metrics --tile_overlap 0.9
# Training command that was used (same as above but on subtrain, using checkpoint-best.pth), can also be launched on multiple gpus
python -u stereoflow/train.py stereo --criterion "LaplacianLossBounded2()" --dataset "CREStereo('train')+SceneFlow('train_allpass')+30*ETH3DLowRes('subtrain')+50*Md05('subtrain')+50*Md06('subtrain')+50*Md14('subtrain')+50*Md21('subtrain')+50*MdEval3('subtrain_full')+Booster('subtrain_balanced')" --val_dataset "SceneFlow('test1of100_finalpass')+SceneFlow('test1of100_cleanpass')+ETH3DLowRes('subval')+Md05('subval')+Md06('subval')+Md14('subval')+Md21('subval')+MdEval3('subval_full')+Booster('subval_balanced')" --lr 3e-5 --batch_size 6 --epochs 32 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --output_dir xps/crocostereo/main_subtrain/
```

##### Other models 

<details>
	<summary>Model for ETH3D</summary> 
	The model used for the submission on ETH3D is trained with the same command but using an unbounded Laplacian loss.
	
	# Download the model
	bash stereoflow/download_model.sh crocostereo_eth3d.pth
	# ETH3D submission
	python stereoflow/test.py --model stereoflow_models/crocostereo_eth3d.pth --dataset "ETH3DLowRes('all')" --save submission --tile_overlap 0.9
	# Training command that was used
	python -u stereoflow/train.py stereo --criterion "LaplacianLoss()" --tile_conf_mode conf_expbeta3 --dataset "CREStereo('train')+SceneFlow('train_allpass')+30*ETH3DLowRes('train')+50*Md05('train')+50*Md06('train')+50*Md14('train')+50*Md21('train')+50*MdEval3('train_full')+Booster('train_balanced')" --val_dataset "SceneFlow('test1of100_finalpass')+SceneFlow('test1of100_cleanpass')+ETH3DLowRes('subval')+Md05('subval')+Md06('subval')+Md14('subval')+Md21('subval')+MdEval3('subval_full')+Booster('subval_balanced')" --lr 3e-5 --batch_size 6 --epochs 32 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --output_dir xps/crocostereo/main_eth3d/
	
</details>

<details>
	<summary>Main model finetuned on Kitti</summary>
	
	# Download the model
	bash stereoflow/download_model.sh crocostereo_finetune_kitti.pth
	# Kitti submission 
	python stereoflow/test.py --model stereoflow_models/crocostereo_finetune_kitti.pth --dataset "Kitti15('test')" --save submission --tile_overlap 0.9
	# Training that was used
	python -u stereoflow/train.py stereo --crop 352 1216 --criterion "LaplacianLossBounded2()" --dataset "Kitti12('train')+Kitti15('train')" --lr 3e-5 --batch_size 1 --accum_iter 6 --epochs 20 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --start_from stereoflow_models/crocostereo.pth --output_dir xps/crocostereo/finetune_kitti/ --save_every 5
</details>

<details>
	<summary>Main model finetuned on Spring</summary>
	
	# Download the model
	bash stereoflow/download_model.sh crocostereo_finetune_spring.pth
	# Spring submission 
	python stereoflow/test.py --model stereoflow_models/crocostereo_finetune_spring.pth --dataset "Spring('test')" --save submission --tile_overlap 0.9
	# Training command that was used
	python -u stereoflow/train.py stereo --criterion "LaplacianLossBounded2()" --dataset "Spring('train')" --lr 3e-5 --batch_size 6 --epochs 8 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --start_from stereoflow_models/crocostereo.pth --output_dir xps/crocostereo/finetune_spring/
</details>

<details>
	<summary>Smaller models</summary>
	To train CroCo-Stereo with smaller CroCo pretrained models, simply replace the <code>--pretrained</code> argument. To download the smaller CroCo-Stereo models based on CroCo v2 pretraining with ViT-Base encoder and Small encoder, use <code>bash stereoflow/download_model.sh crocostereo_subtrain_vitb_smalldecoder.pth</code>, and for the model with a ViT-Base encoder and a Base decoder, use <code>bash stereoflow/download_model.sh crocostereo_subtrain_vitb_basedecoder.pth</code>.
</details>
	

### CroCo-Flow

##### Main model

The main training of CroCo-Flow was performed on the FlyingThings, FlyingChairs, MPI-Sintel and TartanAir datasets.
It was used for our submission to the MPI-Sintel benchmark.

```
# Download the model 
bash stereoflow/download_model.sh crocoflow.pth
# Evaluation 
python stereoflow/test.py --model stereoflow_models/crocoflow.pth --dataset "MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')" --save metrics --tile_overlap 0.9
# Sintel submission
python stereoflow/test.py --model stereoflow_models/crocoflow.pth --dataset "MPISintel('test_allpass')" --save submission --tile_overlap 0.9
# Training command that was used, with checkpoint-best.pth
python -u stereoflow/train.py flow --criterion "LaplacianLossBounded()" --dataset "40*MPISintel('subtrain_cleanpass')+40*MPISintel('subtrain_finalpass')+4*FlyingThings('train_allpass')+4*FlyingChairs('train')+TartanAir('train')" --val_dataset "MPISintel('subval_cleanpass')+MPISintel('subval_finalpass')" --lr 2e-5 --batch_size 8 --epochs 240 --img_per_epoch 30000 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --output_dir xps/crocoflow/main/
```

##### Other models 

<details>
	<summary>Main model finetuned on Kitti</summary>
	
	# Download the model
	bash stereoflow/download_model.sh crocoflow_finetune_kitti.pth
	# Kitti submission 
	python stereoflow/test.py --model stereoflow_models/crocoflow_finetune_kitti.pth --dataset "Kitti15('test')" --save submission --tile_overlap 0.99
	# Training that was used, with checkpoint-last.pth
	python -u stereoflow/train.py flow --crop 352 1216 --criterion "LaplacianLossBounded()" --dataset "Kitti15('train')+Kitti12('train')" --lr 2e-5 --batch_size 1 --accum_iter 8 --epochs 150 --save_every 5 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --start_from stereoflow_models/crocoflow.pth --output_dir xps/crocoflow/finetune_kitti/
</details>

<details>
	<summary>Main model finetuned on Spring</summary>
	
	# Download the model
	bash stereoflow/download_model.sh crocoflow_finetune_spring.pth
	# Spring submission 
	python stereoflow/test.py --model stereoflow_models/crocoflow_finetune_spring.pth --dataset "Spring('test')" --save submission --tile_overlap 0.9
	# Training command that was used, with checkpoint-last.pth
	python -u stereoflow/train.py flow --criterion "LaplacianLossBounded()" --dataset "Spring('train')" --lr 2e-5 --batch_size 8 --epochs 12 --pretrained pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth --start_from stereoflow_models/crocoflow.pth --output_dir xps/crocoflow/finetune_spring/
</details>

<details>
	<summary>Smaller models</summary>
	To train CroCo-Flow with smaller CroCo pretrained models, simply replace the <code>--pretrained</code> argument. To download the smaller CroCo-Flow models based on CroCo v2 pretraining with ViT-Base encoder and Small encoder, use <code>bash stereoflow/download_model.sh crocoflow_vitb_smalldecoder.pth</code>, and for the model with a ViT-Base encoder and a Base decoder, use <code>bash stereoflow/download_model.sh crocoflow_vitb_basedecoder.pth</code>.
</details>
