ARG BASE=dustynv/pytorch:2.7-r36.4.0-cu128-24.04
FROM ${BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://pypi.org/simple/ -r /tmp/requirements.txt

COPY server/ /opt/server/

WORKDIR /opt
ENV PYTHONPATH=/opt

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
