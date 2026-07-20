import base64

import numpy as np

HIDDEN = 8  # conftest.FakeSession의 임베딩 차원


def _int8(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.int8)


def test_single_string_input_returns_one_embedding(client):
    r = client.post("/v1/embeddings", json={"input": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0


def test_default_encoding_is_float(client):
    """업계 표준(OpenAI/Voyage/Cohere/Jina)을 따른다. 공식 pplx 기본값(base64_int8)과는
    의도적으로 다르므로, API 폴백 시에는 encoding_format 을 명시해야 한다."""
    r = client.post("/v1/embeddings", json={"input": "hello"})
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == HIDDEN
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_official_default_still_available_explicitly(client):
    """base64_int8 을 명시하면 공식 API 와 같은 형식을 준다."""
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64_int8"})
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, str)
    assert len(_int8(emb)) == HIDDEN


def test_float_encoding_returns_normalized_array(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "float"})
    emb = r.json()["data"][0]["embedding"]
    assert isinstance(emb, list)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_base64_float32_encoding(client):
    r = client.post(
        "/v1/embeddings", json={"input": "hello", "encoding_format": "base64_float32"}
    )
    emb = np.frombuffer(base64.b64decode(r.json()["data"][0]["embedding"]), dtype="<f4")
    assert len(emb) == HIDDEN
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)


def test_base64_binary_is_packed(client):
    r = client.post(
        "/v1/embeddings", json={"input": "hello", "encoding_format": "base64_binary"}
    )
    packed = base64.b64decode(r.json()["data"][0]["embedding"])
    assert len(packed) == HIDDEN // 8


def test_dimensions_truncates_output(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 128})
    # FakeSession은 8차원이라 128을 요청해도 슬라이스는 8에서 멈춘다.
    # 여기서 확인하는 것은 dimensions가 거부되지 않고 경로를 탄다는 것.
    assert r.status_code == 200


def test_dimensions_below_minimum_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 64})
    assert r.status_code == 422


def test_dimensions_above_maximum_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 2048})
    assert r.status_code == 422


def test_dimensions_must_be_multiple_of_8(client):
    """base64_binary가 packbits라 8의 배수가 아니면 바이트 경계가 어긋난다."""
    r = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 130})
    assert r.status_code == 422


def test_batch_input_preserves_order_and_index(client):
    texts = ["alpha", "beta gamma", "delta epsilon zeta"]
    r = client.post("/v1/embeddings", json={"input": texts, "encoding_format": "float"})
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1, 2]

    # 개별 호출 결과와 배치 결과가 같아야 한다(패딩이 값을 오염시키지 않음).
    for i, t in enumerate(texts):
        single = client.post(
            "/v1/embeddings", json={"input": t, "encoding_format": "float"}
        ).json()["data"][0]
        assert np.allclose(single["embedding"], data[i]["embedding"], atol=1e-6)


def test_invalid_encoding_format_is_rejected(client):
    r = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "int4"})
    assert r.status_code == 422


def test_usage_counts_non_padding_tokens(client):
    r = client.post("/v1/embeddings", json={"input": ["a b c", "d"]})
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] == 4
    assert usage["total_tokens"] == 4


def test_empty_input_list_is_rejected(client):
    assert client.post("/v1/embeddings", json={"input": []}).status_code == 422


def test_empty_string_is_rejected(client):
    """공식 문서: Empty strings are not allowed."""
    assert client.post("/v1/embeddings", json={"input": ""}).status_code == 422
    assert client.post("/v1/embeddings", json={"input": ["ok", "  "]}).status_code == 422


def test_too_many_inputs_are_rejected(client):
    """공식 문서: Max 512 texts per request."""
    r = client.post("/v1/embeddings", json={"input": ["x"] * 513})
    assert r.status_code == 422


def test_base64_alias_returns_float32(client):
    """표준 OpenAI SDK 가 보내는 base64는 float32로 해석되어야 한다."""
    a = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64"})
    b = client.post(
        "/v1/embeddings", json={"input": "hello", "encoding_format": "base64_float32"}
    )
    assert a.json()["data"][0]["embedding"] == b.json()["data"][0]["embedding"]
    # base64 별칭은 float32 이므로 int8 과 달라야 한다.
    c = client.post("/v1/embeddings", json={"input": "hello", "encoding_format": "base64_int8"})
    assert a.json()["data"][0]["embedding"] != c.json()["data"][0]["embedding"]


def test_model_name_is_echoed(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.json()["model"] == "perplexity-ai/pplx-embed-v1-0.6b"
