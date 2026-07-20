"""embed()가 요청된 표현에 필요한 출력만 그래프에서 받아오는지."""

import pytest


@pytest.mark.parametrize(
    "quantization,expected",
    [
        ("float", "pooler_output"),
        ("int8", "pooler_output_int8"),
        ("binary", "pooler_output_binary"),
        ("ubinary", "pooler_output_binary"),
    ],
)
def test_only_required_output_is_requested(fake_embedder, quantization, expected):
    """run(None)으로 전부 받으면 last_hidden_state(batch×seq×1024)가 딸려온다."""
    fake_embedder.embed(["hello"], quantization)
    call = fake_embedder._session.calls[-1]
    assert call["output_names"] == [expected]


def test_last_hidden_state_is_never_requested(fake_embedder):
    for q in ("float", "int8", "binary", "ubinary"):
        fake_embedder.embed(["hello"], q)
    requested = {n for c in fake_embedder._session.calls for n in c["output_names"]}
    assert "last_hidden_state" not in requested


def test_unknown_quantization_raises_before_inference(fake_embedder):
    with pytest.raises(ValueError, match="int4"):
        fake_embedder.embed(["hello"], "int4")
    assert fake_embedder._session.calls == []


def test_token_count_excludes_padding(fake_embedder):
    _, tokens = fake_embedder.embed(["a b c", "d"])
    assert tokens == 4
