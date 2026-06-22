ARG AIRFLOW_TAG

FROM apache/airflow:${AIRFLOW_TAG}

RUN pip install --no-cache-dir --upgrade \
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
    apache-airflow-providers-trino
