def test_single_document_returns_one_result(client):
    r = client.post("/v1/rerank", json={"query": "hello world", "documents": ["doc one"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["results"]) == 1
    assert body["results"][0]["index"] == 0
    assert isinstance(body["results"][0]["relevance_score"], float)


def test_multiple_documents_preserves_order_by_score(client):
    r = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["a", "b c", "d e f"]},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    # Scores should be sorted descending
    scores = [res["relevance_score"] for res in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_top_n_limits_results(client):
    r = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["a", "b", "c", "d", "e"], "top_n": 2},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_top_n_returns_highest_scores(client):
    r = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["a", "b c", "d e f"], "top_n": 2},
    )
    results = r.json()["results"]
    assert len(results) == 2
    # The two highest-scoring documents
    assert results[0]["relevance_score"] >= results[1]["relevance_score"]


def test_model_name_is_echoed(client):
    r = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["doc"], "model": "my-model"},
    )
    assert r.json()["model"] == "my-model"


def test_default_model_name(client):
    r = client.post("/v1/rerank", json={"query": "test", "documents": ["doc"]})
    assert r.json()["model"] == "cstr/llama-nemotron-rerank-1b-v2-ONNX"


def test_empty_query_is_rejected(client):
    r = client.post("/v1/rerank", json={"query": "", "documents": ["doc"]})
    assert r.status_code == 422


def test_empty_documents_list_is_rejected(client):
    r = client.post("/v1/rerank", json={"query": "test", "documents": []})
    assert r.status_code == 422


def test_empty_document_string_is_rejected(client):
    r = client.post("/v1/rerank", json={"query": "test", "documents": ["  "]})
    assert r.status_code == 422


def test_index_reflects_original_position(client):
    r = client.post(
        "/v1/rerank",
        json={"query": "test", "documents": ["first", "second", "third"]},
    )
    indices = {res["index"] for res in r.json()["results"]}
    assert indices == {0, 1, 2}
