ARG BASE=dustynv/vllm:r36.4-cu129-24.04
FROM ${BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Base image already ships the Jetson-built torch 2.8 / triton 3.4 /
# bitsandbytes / accelerate. Flux2 classes need diffusers >=0.38, and sdnq is
# not bundled — install only those on top.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ -r /tmp/requirements.txt

COPY server/ /opt/server/

WORKDIR /opt
ENV PYTHONPATH=/opt

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
