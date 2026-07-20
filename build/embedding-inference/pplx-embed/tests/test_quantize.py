import numpy as np
import pytest

from server.embedder import OUTPUT_FOR, output_name, postprocess

# tanh 전 pooled 값. 부호가 섞이도록 구성.
POOLED = np.array(
    [
        [0.5, -0.25, 0.0, 2.0],
        [-1.5, 0.75, -0.05, 0.1],
    ],
    dtype=np.float32,
)


def test_output_name_maps_each_quantization():
    assert output_name("float") == "pooler_output"
    assert output_name("int8") == "pooler_output_int8"
    assert output_name("binary") == "pooler_output_binary"
    # ubinary는 그래프 출력이 없어 binary에서 유도한다.
    assert output_name("ubinary") == "pooler_output_binary"


def test_output_name_rejects_unknown():
    with pytest.raises(ValueError, match="int4"):
        output_name("int4")


def test_float_is_l2_normalized_tanh():
    out = postprocess(POOLED, "float")
    expected = np.tanh(POOLED)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    assert out.dtype == np.float32
    assert np.allclose(out, expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-6)


def test_float_preserves_direction_of_tanh():
    """L2 정규화는 방향을 바꾸지 않는다 — tanh 결과와 코사인이 1이어야 한다."""
    out = postprocess(POOLED, "float")
    t = np.tanh(POOLED)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    assert np.allclose((out * t).sum(axis=-1), 1.0, atol=1e-6)


def test_int8_passes_graph_output_through_untouched():
    """공식 출력을 그대로 내보내야 한다 — 우리가 재계산하지 않는다."""
    graph_int8 = np.array([[85, -72, 0, 12], [-100, 60, -6, 13]], dtype=np.int8)
    out = postprocess(graph_int8, "int8")
    assert out is graph_int8


def test_binary_passes_graph_output_through_untouched():
    graph_binary = np.array([[1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, -1.0, 1.0]], dtype=np.float32)
    out = postprocess(graph_binary, "binary")
    assert out is graph_binary


def test_ubinary_packs_binary_output():
    """공식 PackedBinaryQuantizer: packbits(x >= 0).

    binary 출력이 x>=0에서 +1이므로 packbits(binary > 0)과 같다.
    """
    graph_binary = np.where(POOLED >= 0, 1.0, -1.0).astype(np.float32)
    out = postprocess(graph_binary, "ubinary")
    assert out.dtype == np.uint8
    assert out.shape == (2, 1)  # 4차원 → 1바이트
    assert np.array_equal(out, np.packbits(POOLED >= 0, axis=-1))


def test_postprocess_rejects_unknown():
    with pytest.raises(ValueError, match="int4"):
        postprocess(POOLED, "int4")


def test_zero_vector_does_not_produce_nan():
    """전부 0인 벡터를 정규화할 때 0으로 나누지 않아야 한다."""
    out = postprocess(np.zeros((1, 4), dtype=np.float32), "float")
    assert not np.isnan(out).any()


def test_all_schema_quantizations_are_mapped():
    """schemas.Quantization과 OUTPUT_FOR가 어긋나지 않도록."""
    from typing import get_args

    from server.schemas import Quantization

    assert set(get_args(Quantization)) == set(OUTPUT_FOR)
