FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ARG FORCE_CUDA=1
ENV FORCE_CUDA=${FORCE_CUDA}

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
    
RUN apt-get update -y && apt-get install -y git wget

WORKDIR /
RUN git clone --recursive https://github.com/NVlabs/InstantSplat.git &&\
    cd InstantSplat &&\
    git submodule update --init --recursive &&\
    cd submodules/dust3r/ &&\
    mkdir -p checkpoints/ &&\
    wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

WORKDIR /InstantSplat

RUN pip3 install -r requirements.txt

ARG TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN pip3 install submodules/simple-knn
RUN sed -i 's/p_view.z <= 0.2f/p_view.z <= 0.001f/g' /InstantSplat/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
RUN pip3 install submodules/diff-gaussian-rasterization

RUN cd submodules/dust3r/croco/models/curope/ &&\
    python3 setup.py build_ext --inplace
