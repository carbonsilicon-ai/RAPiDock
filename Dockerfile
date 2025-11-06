# basic image
FROM m.daocloud.io/docker.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
             echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
             echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
             echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
             echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
             # 可选：添加源码源（若不需要编译源码可删除）
             echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
             echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
             echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
             echo "deb-src http://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list


RUN apt-get update && apt-get install -y wget bzip2 pkg-config build-essential python3-dev python3-pip libatlas-base-dev gfortran libfreetype6-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.11.1-0-Linux-x86_64.sh && \
	bash Miniconda3-py39_24.11.1-0-Linux-x86_64.sh -b -f -p /opt/miniconda && \
    rm Miniconda3-py39_24.11.1-0-Linux-x86_64.sh

ENV PATH=/opt/miniconda/bin:$PATH

RUN conda init bash

COPY pyrosetta-2025.33+release.a492b89a9e-cp39-cp39-linux_x86_64.whl /tmp/pyrosetta-2025.33+release.a492b89a9e-cp39-cp39-linux_x86_64.whl
COPY rapidock_env.yaml /tmp/rapidock_env.yaml
COPY requirement.txt /tmp/requirement.txt

RUN conda env create -f /tmp/rapidock_env.yaml -n RAPiDock && echo "conda activate RAPiDock" >> ~/.bashrc

ENV PATH=/opt/miniconda/envs/RAPiDock/bin:$PATH

RUN /opt/miniconda/envs/RAPiDock/bin/pip install --no-cache-dir -r /tmp/requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN /opt/miniconda/envs/RAPiDock/bin/python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

RUN /opt/miniconda/envs/RAPiDock/bin/pip install /tmp/pyrosetta-2025.33+release.a492b89a9e-cp39-cp39-linux_x86_64.whl

#RUN /opt/miniconda/envs/RAPiDock/bin/python inference.py --config default_inference_args.yaml --protein_peptide_csv data/protein_peptide_example.csv --output_dir results/default

# -v /home/luohao/codes/RAPiDock/esm2_t33_650M_UR50D-contact-regression.pt:/root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt   -v /home/luohao/codes/RAPiDock/esm2_t33_650M_UR50D.pt:/root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
COPY esm2_t33_650M_UR50D-contact-regression.pt /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt
COPY esm2_t33_650M_UR50D.pt /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt

WORKDIR /workdir


