import numpy as np
import pytest
import torch

HIDDEN = 2048  # Nemotron-3-Embed-1B hidden dimension


class FakeOutputs:
    """Mock transformers model output with last_hidden_state."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel:
    """Mock AutoModel that returns deterministic embeddings."""

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def __call__(self, input_ids, attention_mask):
        batch, seq = input_ids.shape
        # Deterministic: sum of token ids as seed
        seeds = input_ids.sum(dim=-1, dtype=torch.float32).reshape(batch, 1)
        hidden = torch.arange(HIDDEN, dtype=torch.float32).reshape(1, HIDDEN) * 0.01
        hidden = hidden.expand(batch, HIDDEN) + seeds * 1e-4
        return FakeOutputs(last_hidden_state=hidden.unsqueeze(1).expand(batch, seq, HIDDEN).contiguous())


class FakeTokenizer:
    """Mock tokenizer. Each word becomes one token id."""

    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, texts, **kwargs):
        batch = []
        for t in texts:
            ids = [len(w) for w in t.split()]
            batch.append(ids)
        max_len = max(len(r) for r in batch) if batch else 1
        padded = [r + [0] * (max_len - len(r)) for r in batch]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * len(r) + [0] * (max_len - len(r)) for r in batch], dtype=torch.long
            ),
        }


@pytest.fixture
def fake_embedder():
    from server.embedder import NemotronEmbedder

    return NemotronEmbedder(model=FakeModel(), tokenizer=FakeTokenizer())


@pytest.fixture
def client(fake_embedder, monkeypatch):
    monkeypatch.setenv("MODEL_ID", "nvidia/Nemotron-3-Embed-1B-BF16")
    from server import app as app_module

    app_module.EMBEDDER_LOADER = lambda: fake_embedder
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
