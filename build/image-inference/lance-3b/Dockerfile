ARG BASE=dustynv/vllm:r36.4-cu129-24.04
FROM ${BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Base image already ships the Jetson-built torch 2.8 / torchvision / torchaudio
# / triton 3.4 / flash_attn 2.8 / transformers 4.53.1 / accelerate / diffusers /
# safetensors / bitsandbytes / torchao / xformers. We only add pure-python deps.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ -r /tmp/requirements.txt

# Vendor Lance source + the AWQ runtime toolkit at pinned commits. Neither repo
# is a pip package, so we clone + put them on PYTHONPATH (see server/pipeline.py
# which spawns lance-quant/comfyui/lance_worker.py as a subprocess).
ARG LANCE_COMMIT=3c757d257eecf9b88b83124c63883038bafccf72
ARG LANCE_QUANT_COMMIT=4edb84856b57beb0956fcd0da59e2b7b6bf55aff
RUN git clone https://github.com/bytedance/Lance.git /opt/Lance \
    && git -C /opt/Lance checkout ${LANCE_COMMIT} \
    && git clone https://github.com/Reza2kn/lance-quant.git /opt/lance-quant \
    && git -C /opt/lance-quant checkout ${LANCE_QUANT_COMMIT}

# Import-only stub for decord (no aarch64 wheel; only video tasks use it).
COPY stubs/ /opt/stubs/

COPY server/ /opt/server/

WORKDIR /opt
ENV PYTHONPATH=/opt:/opt/stubs
# Disable torch.compile / dynamo: Lance imports torch.compile(flex_attention) at
# module load, but inference uses flash_attn_varlen_func (not flex_attention), so
# compilation is dead weight and we avoid any Triton compile path on Jetson.
ENV TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
