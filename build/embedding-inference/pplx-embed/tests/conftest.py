import numpy as np
import pytest

HIDDEN = 8


class FakeOutput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """onnxruntime.InferenceSession 대역. 실제 모델 없이 앱을 돌리기 위한 것.

    pooler_output은 입력 토큰 id 합에서 결정적으로 만들어 낸다 — 같은 입력이면
    같은 벡터가 나오므로 테스트에서 비교가 가능하다.
    """

    OUTPUT_NAMES = [
        "last_hidden_state",
        "pooler_output",
        "pooler_output_int8",
        "pooler_output_binary",
    ]

    def __init__(self):
        self.calls = []

    def get_outputs(self):
        return [FakeOutput(n) for n in self.OUTPUT_NAMES]

    def run(self, output_names, feeds):
        ids = feeds["input_ids"]
        self.calls.append({"output_names": output_names, "feeds": feeds})
        batch, seq = ids.shape
        seeds = ids.sum(axis=-1).astype(np.float32).reshape(batch, 1)
        pooled = np.tanh(
            np.arange(HIDDEN, dtype=np.float32).reshape(1, HIDDEN) * 0.1 + seeds * 1e-4
        )
        available = {
            "last_hidden_state": np.zeros((batch, seq, HIDDEN), dtype=np.float32),
            "pooler_output": pooled,
            "pooler_output_int8": np.round(np.tanh(pooled) * 127).astype(np.int8),
            "pooler_output_binary": np.where(pooled >= 0, 1.0, -1.0).astype(np.float32),
        }
        names = output_names if output_names is not None else self.OUTPUT_NAMES
        return [available[n] for n in names]


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1 if i else 0 for i in ids]


class FakeTokenizer:
    """tokenizers.Tokenizer 대역. 단어 수만큼 토큰을 만들고 패딩한다."""

    def __init__(self, max_length=2048):
        self.max_length = max_length
        self.truncation = None
        self.padding = None

    def enable_truncation(self, max_length):
        self.truncation = max_length

    def enable_padding(self, **kwargs):
        self.padding = kwargs

    def encode_batch(self, texts):
        raw = [[len(w) for w in t.split()][: self.truncation or self.max_length] for t in texts]
        width = max((len(r) for r in raw), default=1) or 1
        return [FakeEncoding(r + [0] * (width - len(r))) for r in raw]


@pytest.fixture
def fake_embedder():
    from server.embedder import PplxEmbedder

    return PplxEmbedder(session=FakeSession(), tokenizer=FakeTokenizer())


@pytest.fixture
def client(fake_embedder, monkeypatch):
    monkeypatch.setenv("MODEL_ID", "perplexity-ai/pplx-embed-v1-0.6b")
    from server import app as app_module

    app_module.EMBEDDER_LOADER = lambda: fake_embedder  # injection seam
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
