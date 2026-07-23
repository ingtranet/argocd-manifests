# llama.cpp server for Jetson Orin (L4T r36.4, CUDA 12.8, sm_87).
# Built on the Orin node (aarch64) and pushed to
# zot.ingtra.net/library/image-inference — see build.sh.
#
# This image includes Python 3 + numpy + safetensors + FastAPI/uvicorn
# for the Jina v3 reranker HTTP wrapper (POST /v1/rerank).
ARG BASE=dustynv/cuda:12.8-samples-r36.4.0-cu128-24.04

FROM ${BASE} AS build

# Pin llama.cpp to an upstream release tag. b10099 is the latest as of 2026-07-24.
ARG LLAMA_TAG=b10103

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build libcurl4-openssl-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggml-org/llama.cpp.git \
 && cd llama.cpp \
 && git checkout "tags/${LLAMA_TAG}" \
 && git submodule update --init --recursive

WORKDIR /build/llama.cpp
RUN cmake -B build \
        -DGGML_CUDA=on \
        -DGGML_CUDA_F16=on \
        -DLLAMA_CURL=on \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="87" \
        -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TESTS=OFF \
 && cmake --build build --config Release \
 && cmake --install build

FROM ${BASE}

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends libcurl4 python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install Python runtime deps for Jina v3 reranker wrapper
RUN pip3 install --no-cache-dir -i https://pypi.org/simple/ numpy safetensors fastapi uvicorn

COPY --from=build /opt/llama.cpp/ /usr/local/

# Copy the Jina reranker HTTP wrapper
COPY jina-reranker-wrapper.py /usr/local/bin/jina-reranker-wrapper.py
RUN chmod +x /usr/local/bin/jina-reranker-wrapper.py

EXPOSE 8080

# Default entrypoint is llama-server; override with jina-reranker-wrapper.py for Jina.
ENTRYPOINT ["/usr/local/bin/llama-server"]
