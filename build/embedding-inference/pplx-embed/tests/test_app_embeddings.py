import numpy as np


def test_single_string_input_returns_one_embedding(client):
    r = client.post("/v1/embeddings", json={"input": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 8


def test_batch_input_preserves_order_and_index(client):
    texts = ["alpha", "beta gamma", "delta epsilon zeta"]
    r = client.post("/v1/embeddings", json={"input": texts})
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2]

    # 개별 호출 결과와 배치 결과가 같아야 한다(패딩이 값을 오염시키지 않음).
    for i, t in enumerate(texts):
        single = client.post("/v1/embeddings", json={"input": t}).json()["data"][0]
        assert np.allclose(single["embedding"], data[i]["embedding"], atol=1e-6)


def test_default_quantization_is_normalized_float(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    emb = np.array(r.json()["data"][0]["embedding"])
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_int8_quantization_returns_integers_in_range(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "quantization": "int8"})
    emb = r.json()["data"][0]["embedding"]
    assert all(float(v).is_integer() for v in emb)
    assert all(-128 <= v <= 127 for v in emb)


def test_ubinary_quantization_packs_dimensions(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "quantization": "ubinary"})
    emb = r.json()["data"][0]["embedding"]
    assert len(emb) == 1  # 8차원 → 1바이트
    assert all(0 <= v <= 255 for v in emb)


def test_invalid_quantization_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "quantization": "int4"})
    assert r.status_code == 422


def test_usage_counts_non_padding_tokens(client):
    r = client.post("/v1/embeddings", json={"input": ["a b c", "d"]})
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] == 4
    assert usage["total_tokens"] == 4


def test_empty_input_list_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": []})
    assert r.status_code == 422


def test_model_name_is_echoed(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.json()["model"] == "perplexity-ai/pplx-embed-v1-0.6b"
