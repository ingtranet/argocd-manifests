# BF16 embedding server for nvidia/Nemotron-3-Embed-1B-BF16.
#
# CUDA-only — BF16 inference requires CUDA. No CPU fallback.
# Uses HuggingFace transformers with torch.bfloat16.
#
# ⚠  BF16 is NOT INT8. This model uses bfloat16 precision weights.
#    The existing INT8 ONNX pipeline (pplx-embed) is a different format.
#    These are incompatible and cannot share the same serving infrastructure.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY server/ /opt/server/

WORKDIR /opt
ENV PYTHONPATH=/opt

EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
