import base64

import numpy as np

HIDDEN = 2048  # conftest.FakeModel hidden dimension


def test_single_string_input_returns_one_embedding(client):
    r = client.post("/v1/embeddings", json={"input": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0


def test_default_encoding_is_float(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == HIDDEN
    # L2 normalized
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_float_encoding_returns_normalized_array(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "float"})
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_base64_encoding(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64"})
    emb = np.frombuffer(base64.b64decode(r.json()["data"][0]["embedding"]), dtype="<f4")
    assert len(emb) == HIDDEN
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_batch_input_preserves_order_and_index(client):
    texts = ["alpha", "beta gamma", "delta epsilon zeta"]
    r = client.post("/v1/embeddings", json={"input": texts, "encoding_format": "float"})
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2]

    for i, t in enumerate(texts):
        single = client.post(
            "/v1/embeddings", json={"input": t, "encoding_format": "float"}
        ).json()["data"][0]
        assert np.allclose(single["embedding"], data[i]["embedding"], atol=1e-6)


def test_invalid_encoding_format_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64_int8"})
    assert r.status_code == 422


def test_empty_input_list_is_rejected(client):
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 422


def test_empty_string_is_rejected(client):
    assert client.post("/v1/embeddings", json={"input": ""}).status_code == 422
    assert client.post("/v1/embeddings", json={"input": ["ok", "  "]}).status_code == 422


def test_too_many_inputs_are_rejected(client):
    r = client.post("/v1/embeddings", json={"input": ["x"] * 513})
    assert r.status_code == 422


def test_model_name_is_echoed(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.json()["model"] == "nvidia/Nemotron-3-Embed-1B-BF16"


def test_dimensions_truncates_output(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 128})
    assert r.status_code == 200
    emb = r.json()["data"][0]["embedding"]
    assert len(emb) == 128
    # Re-normalized after truncation
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_dimensions_above_max_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 4096})
    assert r.status_code == 422
