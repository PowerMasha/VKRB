ARG CUDA_VERSION=11.1

# onnxruntime-gpu requires cudnn
ARG CUDNN_VER=8

# See possible types: https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated
ARG IMAGE_TYPE=devel

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VER}-${IMAGE_TYPE}-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt update && \
    apt install -y software-properties-common && \
    # the latest version of Git
    add-apt-repository ppa:git-core/ppa && \
    apt update && \
    apt install -y --no-install-recommends \
    git \
    gosu \
    nano \
    ca-certificates \
    wget \
    unzip \
    libgl1-mesa-glx \
    build-essential \ 
    && \
    apt autoremove -y && \
    apt clean -y

ARG CMAKE_VERSION=3.19.6

# Install CMake
RUN wget -q -O ./cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    mkdir /opt/cmake && \
    sh ./cmake.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm ./cmake.sh

# Install Miniconda See possible versions: https://repo.anaconda.com/miniconda/
ARG CONDA_VERSION=latest

ARG CONDA_DIR=/opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q -O ./miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh \
    && sh ./miniconda.sh -bfp $CONDA_DIR \
    && rm ./miniconda.sh

WORKDIR /home

COPY ./environment.yml ./environment.yml 

RUN conda env update -n base --prune --file ./environment.yml && \
    conda clean -ayf && rm ./environment.yml

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgtk2.0-dev gosu && \
    pip install --no-cache-dir opencv-python==4.5.3.56

ARG SIMSWAP_COMMIT_HASH=926268bc4403133c11177e80e51df3dd5e1ee8fc

RUN git clone --single-branch https://github.com/neuralchen/SimSwap.git && \
    cd ./SimSwap && \
    git checkout ${SIMSWAP_COMMIT_HASH} && \
    rm -rf .git && \
    mkdir checkpoints && \
    mkdir insightface_func/models && \
    mkdir parsing_model/checkpoint && \
    mkdir arcface_model

WORKDIR /home

ARG GPEN_COMMIT_HASH=7f0b3817734094c91177979b81627c01a3f0ea96

RUN git clone --single-branch https://github.com/yangxy/GPEN.git && \
    cd ./GPEN && \
    git checkout ${GPEN_COMMIT_HASH} && \
    rm -rf .git

# ARG MAX_GCC_VERSION=10

# RUN apt install -y gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION

WORKDIR /home/GPEN

# RUN wget https://github.com/ninja-build/ninja/releases/download/v1.10.2/ninja-linux.zip
# RUN unzip ninja-linux.zip -d /usr/local/bin/
# RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

WORKDIR /home/

RUN mkdir input output

COPY ./set_lib.py ./set_lib.py
RUN python ./set_lib.py
RUN rm ./set_lib.py

COPY ./entrypoint.sh ./run.sh ./run_simswap.py ./run_simswap_specific.py ./

RUN chmod o+x ./entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]