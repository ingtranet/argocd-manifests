#!/usr/bin/env bash
# Build the llama.cpp Jetson (sm_87) server image and push it to the internal
# registry. Produces a linux/arm64 image, so run this on an aarch64 host — the
# Orin node, or an Apple Silicon Mac with Docker Desktop.
#
# Tag convention (see ../README.md): <model-name>-<short-git-sha>
set -euo pipefail

cd "$(dirname "$0")"

LLAMA_TAG="${LLAMA_TAG:-f89d448bc44216fe007b25a94f025aea2e93be7e}"
SHA="$(git rev-parse --short HEAD)"
IMAGE="zot.ingtra.net/library/image-inference:llamacpp-${LLAMA_TAG}-${SHA}"

docker build \
    --platform linux/arm64 \
    --build-arg LLAMA_TAG="${LLAMA_TAG}" \
    -t "${IMAGE}" .

docker push "${IMAGE}"

echo "pushed ${IMAGE}"
