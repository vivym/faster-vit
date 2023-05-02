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
    faster-vit-dev bash
