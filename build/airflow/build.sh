#!/bin/bash

AIRFLOW_VERSION=3.2.2
PYTHON_VERSION=3.13

TAG=${AIRFLOW_VERSION}-python${PYTHON_VERSION}
DATE=$(date +%y%m%d)
if command -v openssl >/dev/null 2>&1; then
  RANDOM_STR=$(openssl rand -base64 64 | tr -dc 'a-z0-9' | head -c 3)
else
  RANDOM_STR=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 3)
fi

docker buildx build --platform linux/arm64,linux/amd64 \
  --push \
  --build-arg AIRFLOW_TAG=${TAG} \
  -t docker.io/cookieshake/airflow:${AIRFLOW_VERSION}-${DATE}-${RANDOM_STR} .
