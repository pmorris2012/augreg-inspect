FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV LANG C.UTF-8

RUN apt-get update

COPY . /opt/app
WORKDIR /opt/app

RUN pip3 install \
    --use-deprecated=legacy-resolver \
    -r requirements.txt

RUN apt-get -y update && apt-get -y install ffmpeg

WORKDIR /code
