# Use the NVIDIA CUDA *devel* base image (includes nvcc) for Ubuntu 22.04 with CUDA 11.8
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Prevent interactive prompts from apt
ARG DEBIAN_FRONTEND=noninteractive

# Set a working directory
WORKDIR /app

# 1) Install system packages + Python 3.8 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget git bzip2 vim \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    freeglut3-dev \
    libosmesa6-dev \
    patchelf \  
    x11-apps \
    libedit-dev \
    libffi-dev \
    libncurses-dev \
    libssl-dev \
    tk-dev \
    zlib1g-dev \
    xz-utils \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

# 2) Make python3 -> python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 3) Install pip for Python 3.8, then upgrade
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip setuptools wheel

# Optional: If you really need pinned pip/setuptools/wheel versions:
RUN python3 -m pip install \
    pip \
    setuptools \
    wheel

# 4) Install other Python dependencies (using Python 3.8 now)
RUN apt-get remove -y python3-blinker
RUN python3 -m pip install \
    numpy==1.23.5 \
    trimesh \
    autolab_core \
    h5py \
    pyrender \
    pyglet==1.5.27 \
    python-fcl \
    open3d \
    pybullet \
    scipy \
    pytorch-kinematics \
    ipywidgets

# 5) Install PyTorch, Torchvision, Torchaudio for CUDA 11.8
RUN python3 -m pip install \
    torchvision==0.17.0+cu118 \
    torchaudio==2.2.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 6) Clone your repository
RUN git clone https://github.com/yubink2/AssistiveLimbManipulation.git
WORKDIR /app/AssistiveLimbManipulation

# 7) Install the 'csdf' module from the 'resources/csdf' directory
RUN (cd resources/csdf && python3 -m pip install .)

# 8) PyBullet GUI settings
ENV DISPLAY=:0
EXPOSE 8080