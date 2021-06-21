FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

ENV LANG C.UTF-8

RUN apt-get update && apt-get -y dist-upgrade
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-pandas \
    wget

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    jupyterlab jupyter_http_over_ws ipywidgets \
    && jupyter serverextension enable --py jupyter_http_over_ws \
    && jupyter nbextension enable --py widgetsnbextension

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    numpy \
    scipy \
    matplotlib \
    ipython \
    jupyter \
    sympy \
    nose \
    tqdm \
    scikit-learn \ 
    h5py \
    ftfy \
    regex \
    click \ 
    opencv-python

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    torch==1.8.1+cu111 \
    torchvision==0.9.1+cu111 \
    torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get -y update && apt-get -y install ffmpeg

WORKDIR /code
