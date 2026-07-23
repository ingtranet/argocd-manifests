import numpy as np
import pytest

HIDDEN = 2  # logits dimension for reranker (non-relevant, relevant)


class FakeSession:
    """onnxruntime.InferenceSession stand-in for testing without a real model."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        class FakeInput:
            def __init__(self, name):
                self.name = name
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def run(self, output_names, feeds):
        self.calls.append({"output_names": output_names, "feeds": feeds})
        batch = feeds["input_ids"].shape[0]
        # Deterministic scores: higher for documents with more tokens
        scores = np.array(
            [[-1.0, 1.0 + feeds["input_ids"][i].sum() * 1e-4] for i in range(batch)],
            dtype=np.float32,
        )
        return [scores]


class FakeEncoding:
    def __init__(self, ids, type_ids=None):
        self.ids = ids
        self.attention_mask = [1 if i else 0 for i in ids]
        self.type_ids = type_ids or [0] * len(ids)


class FakeTokenizer:
    """tokenizers.Tokenizer stand-in."""

    def __init__(self, max_length=4096):
        self.max_length = max_length
        self.truncation = None

    def enable_truncation(self, max_length):
        self.truncation = max_length

    def enable_padding(self, **kwargs):
        pass

    def encode_batch(self, pairs):
        # pairs is list of (query, document) tuples
        raw = []
        for query, doc in pairs:
            # Simulate tokenization: each word is a token
            tokens = query.split() + doc.split()
            if self.truncation:
                tokens = tokens[: self.truncation]
            ids = [len(w) for w in tokens]
            raw.append(ids)
        width = max((len(r) for r in raw), default=1) or 1
        return [
            FakeEncoding(r + [0] * (width - len(r)), type_ids=[0] * width)
            for r in raw
        ]


@pytest.fixture
def fake_reranker():
    from server.reranker import NemotronReranker

    return NemotronReranker(session=FakeSession(), tokenizer=FakeTokenizer())


@pytest.fixture
def client(fake_reranker, monkeypatch):
    monkeypatch.setenv("MODEL_ID", "cstr/llama-nemotron-rerank-1b-v2-ONNX")
    from server import app as app_module

    app_module.RERANKER_LOADER = lambda: fake_reranker
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
