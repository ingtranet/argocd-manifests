"""embed()가 요청된 인코딩에 필요한 출력만 그래프에서 받아오는지."""

import pytest


@pytest.mark.parametrize(
    "encoding_format,expected",
    [
        ("float", "pooler_output"),
        ("base64", "pooler_output"),
        ("base64_int8", "pooler_output_int8"),
        ("base64_binary", "pooler_output_binary"),
    ],
)
def test_only_required_output_is_requested(fake_embedder, encoding_format, expected):
    """run(None)으로 전부 받으면 last_hidden_state(batch×seq×1024)가 딸려온다."""
    fake_embedder.embed(["hello"], encoding_format)
    call = fake_embedder._session.calls[-1]
    assert call["output_names"] == [expected]


def test_last_hidden_state_is_never_requested(fake_embedder):
    for f in ("float", "base64", "base64_int8", "base64_binary"):
        fake_embedder.embed(["hello"], f)
    requested = {n for c in fake_embedder._session.calls for n in c["output_names"]}
    assert "last_hidden_state" not in requested


def test_unknown_encoding_format_raises_before_inference(fake_embedder):
    with pytest.raises(ValueError, match="int4"):
        fake_embedder.embed(["hello"], "int4")
    assert fake_embedder._session.calls == []


def test_token_count_excludes_padding(fake_embedder):
    _, tokens = fake_embedder.embed(["a b c", "d"])
    assert tokens == 4


def test_dimensions_truncate_before_encoding(fake_embedder):
    import base64
    import numpy as np

    full, _ = fake_embedder.embed(["hello"], "base64_int8")
    cut, _ = fake_embedder.embed(["hello"], "base64_int8", 4)
    a = np.frombuffer(base64.b64decode(full[0]), dtype=np.int8)
    b = np.frombuffer(base64.b64decode(cut[0]), dtype=np.int8)
    assert len(b) == 4
    assert np.array_equal(b, a[:4])


def test_embedder_default_matches_schema_default():
    """embed()와 API 스키마의 기본 인코딩이 어긋나면 조용한 함정이 된다."""
    from server.embedder import DEFAULT_ENCODING_FORMAT
    from server.schemas import EmbeddingRequest

    assert EmbeddingRequest(input="x").encoding_format == DEFAULT_ENCODING_FORMAT
