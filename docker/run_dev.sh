#!/bin/bash
set -e
set -u

docker run \
    -it \
    --rm \
    --gpus=all \
    "--cap-add=SYS_ADMIN" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PWD}:/workspace/faster-vit \
    --workdir /workspace/faster-vit \
    nvcr.io/nvidia/pytorch:22.08-py3 bash
