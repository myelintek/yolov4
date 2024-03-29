FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV SHELL /bin/bash
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV DEBIAN_FRONTEND=noninteractive
ARG VERSION
ENV VERSION ${VERSION:-dev}

WORKDIR /mlsteam/lab

ADD clean-layer.sh requirements.txt requirements.system install-sshd.sh set_terminal_dark.sh /tmp/

RUN sed -i 's/archive.ubuntu.com/tw.archive.ubuntu.com/g' /etc/apt/sources.list && \
    mkdir -p /mlsteam/data && \
    mkdir -p /mlsteam/lab && \
    apt-get update && \
    xargs apt-get install -y < /tmp/requirements.system && \
    python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    bash /tmp/install-sshd.sh && \
    bash /tmp/set_terminal_dark.sh && \
    bash /tmp/clean-layer.sh

RUN pip3 install --upgrade https://github.com/myelintek/lib-mlsteam/releases/download/v0.3/mlsteam-0.3.0-py3-none-any.whl

ADD src /mlsteam/lab
ADD bash.bashrc /etc/bash.bashrc

RUN cd /mlsteam/lab && make && \
    jupyter nbconvert --to notebook --inplace --execute entry.ipynb

RUN rm -rf /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 /tmp/*

