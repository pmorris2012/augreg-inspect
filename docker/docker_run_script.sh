#!/bin/bash

docker run \
    -it --rm --init \
    --shm-size=16G \
    --gpus all \
    -v /media/mpcrpaul/fastdata/augreg:/data \
    -v $(pwd):/code \
    augreg-inspect python3 /code/"$@"
