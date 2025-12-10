FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel AS base

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget nano unzip ffmpeg libsm6 libxext6 ninja-build cmake build-essential libopenblas-dev \
        xterm xauth openssh-server tmux wget mate-desktop-environment-core && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install matplotlib opencv-python scikit-image timm transformers==4.37.2 numpy==1.24.1
RUN pip install pyyaml requests tqdm omegaconf einops webdataset

RUN pip install -U openmim
RUN mim install mmengine

# Install a compatible version of mmcv-full (1.7.2) for PyTorch 2.1
RUN pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# Install mmsegmentation
RUN pip install mmsegmentation==0.30.0

WORKDIR /home/${USER}/km-vipe

CMD ["/bin/bash"]
