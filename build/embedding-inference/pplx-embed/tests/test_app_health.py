def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "perplexity-ai/pplx-embed-v1-0.6b"


def test_metrics_exposes_embedding_counter(client):
    client.post("/v1/embeddings", json={"input": "hello"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "pplx_embed_embeddings_total" in r.text
