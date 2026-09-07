ARG AIRFLOW_VERSION
ARG PYTHON_VERSION

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    apache-airflow-providers-airbyte \
    apache-airflow-providers-apprise \
    apache-airflow-providers-amazon[s3fs] \
    apache-airflow-providers-apache-kafka \
    apache-airflow-providers-cncf-kubernetes \
    apache-airflow-providers-fab \
    apache-airflow-providers-git \
    apache-airflow-providers-github \
    apache-airflow-providers-google \
    apache-airflow-providers-grpc \
    apache-airflow-providers-http \
    apache-airflow-providers-mongo \
    apache-airflow-providers-openai \
    apache-airflow-providers-postgres \
    apache-airflow-providers-redis \
    apache-airflow-providers-trino \
    --constraint "${HOME}/constraints.txt"
