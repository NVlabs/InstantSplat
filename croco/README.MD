# CroCo + CroCo v2 / CroCo-Stereo / CroCo-Flow

[[`CroCo arXiv`](https://arxiv.org/abs/2210.10716)] [[`CroCo v2 arXiv`](https://arxiv.org/abs/2211.10408)] [[`project page and demo`](https://croco.europe.naverlabs.com/)]

This repository contains the code for our CroCo model presented in our NeurIPS'22 paper [CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion](https://openreview.net/pdf?id=wZEfHUM5ri) and its follow-up extension published at ICCV'23 [Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow](https://openaccess.thecvf.com/content/ICCV2023/html/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.html), refered to as CroCo v2:

![image](assets/arch.jpg)

```bibtex
@inproceedings{croco,
  title={{CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion}},
  author={{Weinzaepfel, Philippe and Leroy, Vincent and Lucas, Thomas and Br\'egier, Romain and Cabon, Yohann and Arora, Vaibhav and Antsfeld, Leonid and Chidlovskii, Boris and Csurka, Gabriela and Revaud J\'er\^ome}},
  booktitle={{NeurIPS}},
  year={2022}
}

@inproceedings{croco_v2,
  title={{CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow}},
  author={Weinzaepfel, Philippe and Lucas, Thomas and Leroy, Vincent and Cabon, Yohann and Arora, Vaibhav and Br{\'e}gier, Romain and Csurka, Gabriela and Antsfeld, Leonid and Chidlovskii, Boris and Revaud, J{\'e}r{\^o}me}, 
  booktitle={ICCV},
  year={2023}
}
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
Some components are based on code from [MAE](https://github.com/facebookresearch/mae) released under the CC BY-NC-SA 4.0 License and [timm](https://github.com/rwightman/pytorch-image-models) released under the Apache 2.0 License.
Some components for stereo matching and optical flow are based on code from [unimatch](https://github.com/autonomousvision/unimatch) released under the MIT license.

## Preparation

1. Install dependencies on a machine with a NVidia GPU using e.g. conda. Note that `habitat-sim` is required only for the interactive demo and the synthetic pre-training data generation. If you don't plan to use it, you can ignore the line installing it and use a more recent python version.

```bash
conda create -n croco python=3.7 cmake=3.14.0
conda activate croco
conda install habitat-sim headless -c conda-forge -c aihabitat
conda install pytorch torchvision -c pytorch
conda install notebook ipykernel matplotlib
conda install ipywidgets widgetsnbextension
conda install scikit-learn tqdm quaternion opencv # only for pretraining / habitat data generation

```

2. Compile cuda kernels for RoPE

CroCo v2 relies on RoPE positional embeddings for which you need to compile some cuda kernels.
```bash
cd models/curope/
python setup.py build_ext --inplace
cd ../../
```

This can be a bit long as we compile for all cuda architectures, feel free to update L9 of `models/curope/setup.py` to compile for specific architectures only.
You might also need to set the environment `CUDA_HOME` in case you use a custom cuda installation.

In case you cannot provide, we also provide a slow pytorch version, which will be automatically loaded.

3. Download pre-trained model

We provide several pre-trained models:

| modelname                                                                                                                          | pre-training data | pos. embed. | Encoder | Decoder |
|------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------|---------|---------|
| [`CroCo.pth`](https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth)                                                 | Habitat           | cosine      | ViT-B   | Small   |
| [`CroCo_V2_ViTBase_SmallDecoder.pth`](https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_SmallDecoder.pth) | Habitat + real    | RoPE        | ViT-B   | Small   |
| [`CroCo_V2_ViTBase_BaseDecoder.pth`](https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth)   | Habitat + real    | RoPE        | ViT-B   | Base    |
| [`CroCo_V2_ViTLarge_BaseDecoder.pth`](https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth) | Habitat + real    | RoPE        | ViT-L   | Base    |

To download a specific model, i.e., the first one (`CroCo.pth`)
```bash
mkdir -p pretrained_models/
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo.pth -P pretrained_models/
```

## Reconstruction example

Simply run after downloading the `CroCo_V2_ViTLarge_BaseDecoder` pretrained model (or update the corresponding line in `demo.py`)
```bash
python demo.py
```

## Interactive demonstration of cross-view completion reconstruction on the Habitat simulator

First download the test scene from Habitat:
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path habitat-sim-data/
```

Then, run the Notebook demo `interactive_demo.ipynb`.

In this demo, you should be able to sample a random reference viewpoint from an [Habitat](https://github.com/facebookresearch/habitat-sim) test scene. Use the sliders to change viewpoint and select a masked target view to reconstruct using CroCo.
![croco_interactive_demo](https://user-images.githubusercontent.com/1822210/200516576-7937bc6a-55f8-49ed-8618-3ddf89433ea4.jpg)

## Pre-training 

### CroCo 

To pre-train CroCo, please first generate the pre-training data from the Habitat simulator, following the instructions in [datasets/habitat_sim/README.MD](datasets/habitat_sim/README.MD) and then run the following command:
```
torchrun --nproc_per_node=4 pretrain.py --output_dir ./output/pretraining/
```

Our CroCo pre-training was launched on a single server with 4 GPUs.
It should take around 10 days with A100 or 15 days with V100 to do the 400 pre-training epochs, but decent performances are obtained earlier in training.
Note that, while the code contains the same scaling rule of the learning rate as MAE when changing the effective batch size, we did not experimented if it is valid in our case.
The first run can take a few minutes to start, to parse all available pre-training pairs.

### CroCo v2 

For CroCo v2 pre-training, in addition to the generation of the pre-training data from the Habitat simulator above, please pre-extract the crops from the real datasets following the instructions in [datasets/crops/README.MD](datasets/crops/README.MD).
Then, run the following command for the largest model (ViT-L encoder, Base decoder):
```
torchrun --nproc_per_node=8 pretrain.py --model "CroCoNet(enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_num_heads=12, dec_depth=12, pos_embed='RoPE100')" --dataset "habitat_release+ARKitScenes+MegaDepth+3DStreetView+IndoorVL" --warmup_epochs 12 --max_epoch 125 --epochs 250 --amp 0 --keep_freq 5 --output_dir ./output/pretraining_crocov2/
```

Our CroCo v2 pre-training was launched on a single server with 8 GPUs for the largest model, and on a single server with 4 GPUs for the smaller ones, keeping a batch size of 64 per gpu in all cases.
The largest model should take around 12 days on A100.
Note that, while the code contains the same scaling rule of the learning rate as MAE when changing the effective batch size, we did not experimented if it is valid in our case.

## Stereo matching and Optical flow downstream tasks

For CroCo-Stereo and CroCo-Flow, please refer to [stereoflow/README.MD](stereoflow/README.MD).
