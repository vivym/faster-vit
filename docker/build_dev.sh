#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

DOCKER_BUILDKIT=1 docker build --build-arg USER=$USER \
                               --build-arg UID=$(id -u) \
                               --build-arg GID=$(id -g) \
                               --build-arg PW=$USER \
                               -t faster-vit-dev \
                               -f docker/Dockerfile.dev \
                               .
