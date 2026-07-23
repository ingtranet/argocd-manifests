#!/usr/bin/env bash
# Nemotron reranker ONNX 서버 이미지를 빌드해 사내 레지스트리에 올린다.
# Jetson ARM64 전용 — linux/arm64로 고정.
#
# 태그 규칙(../README.md): <model-name>-<short-git-sha>
set -euo pipefail

cd "$(dirname "$0")"

SHA="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain -- .)" ]; then
    SHA="${SHA}-dirty${TAG_SUFFIX:+-$TAG_SUFFIX}"
fi
IMAGE="zot.ingtra.net/library/embedding-inference:nemotron-rerank-${SHA}"

docker build --platform linux/arm64 -f Dockerfile.jetson -t "${IMAGE}" .
docker push "${IMAGE}"

echo "pushed ${IMAGE}"
