FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget nano unzip ffmpeg libsm6 libxext6 ninja-build cmake build-essential libopenblas-dev \
        xterm xauth openssh-server tmux wget mate-desktop-environment-core && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

WORKDIR /home/${USER}/
COPY ../envs/ envs/

RUN conda env create -f "envs/base.yml" -n vipe_env 
RUN conda run -n vipe_env python -m pip install -r envs/requirements.txt

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/${USER}/ViPE
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate vipe_env' >> /home/${USER}/.bashrc