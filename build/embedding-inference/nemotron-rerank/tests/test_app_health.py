def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "cstr/llama-nemotron-rerank-1b-v2-ONNX"


def test_metrics_exposes_rerank_counter(client):
    client.post("/v1/rerank", json={"query": "test", "documents": ["doc1"]})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "nemotron_rerank_total" in r.text
