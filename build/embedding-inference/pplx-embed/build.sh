#!/usr/bin/env bash
# pplx-embed ONNX 임베딩 서버 이미지를 빌드해 사내 레지스트리에 올린다.
# 배포 대상이 amd64 노드(k8s-la-worker-2)라 linux/amd64로 고정 — Apple Silicon
# 에서 돌릴 때는 buildx 에뮬레이션이 걸린다.
#
# 태그 규칙(../README.md): <model-name>-<short-git-sha>
set -euo pipefail

cd "$(dirname "$0")"

SHA="$(git rev-parse --short HEAD)"
# 커밋 전 빌드가 커밋된 것과 같은 태그를 덮어쓰지 않도록 표시한다.
# TAG_SUFFIX로 여러 번 굽는 경우를 구분할 수 있다(예: TAG_SUFFIX=2).
# git diff HEAD 는 untracked 파일을 못 본다 — 신규 디렉토리 전체가 untracked인
# 첫 빌드에서 clean으로 오판해 태그를 덮어썼다. status --porcelain 을 쓸 것.
if [ -n "$(git status --porcelain -- .)" ]; then
    SHA="${SHA}-dirty${TAG_SUFFIX:+-$TAG_SUFFIX}"
fi
IMAGE="zot.ingtra.net/library/embedding-inference:pplx-embed-${SHA}"

docker build --platform linux/amd64 -t "${IMAGE}" .
docker push "${IMAGE}"

echo "pushed ${IMAGE}"
