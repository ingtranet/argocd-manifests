# OpenAI 호환 임베딩 서버 for perplexity-ai/pplx-embed-v1-0.6b (int8 ONNX 그래프).
#
# TEI를 쓰지 않는 이유: TEI가 번들한 onnxruntime(ort 2.0.0-rc.10, v1.9.2~main
# 전부 동일 핀)이 이 그래프의 com.microsoft::GatherBlockQuantized `bits` 속성을
# 모른다. fp32 그래프는 로드되지만 warmup 포함 5.3GB를 먹어 노드에서 evict 됐다.
# ORT 1.27은 int8 그래프를 그대로 읽는다.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY server/ /opt/server/

WORKDIR /opt
ENV PYTHONPATH=/opt

EXPOSE 8080
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
