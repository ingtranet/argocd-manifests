import base64

import numpy as np
import pytest

from server.embedder import ALIASES, OUTPUT_FOR, output_name, postprocess, truncate

# tanh 전 pooled 값. 부호가 섞이도록 구성.
POOLED = np.array(
    [
        [0.5, -0.25, 0.0, 2.0, 0.1, -0.9, 0.3, -0.2],
        [-1.5, 0.75, -0.05, 0.1, -0.4, 0.6, -0.7, 0.8],
    ],
    dtype=np.float32,
)


def test_output_name_maps_each_encoding_format():
    assert output_name("float") == "pooler_output"
    assert output_name("base64_float32") == "pooler_output"
    assert output_name("base64_int8") == "pooler_output_int8"
    assert output_name("base64_binary") == "pooler_output_binary"


def test_base64_is_alias_of_base64_float32():
    """공식 enum에 base64는 없다(공식은 422). 표준 OpenAI SDK 가 보내고 float32로
    디코딩하는 값이라 base64_float32에 맞춘다 — int8을 주면 길이 1/4의 쓰레기 값을
    에러 없이 받게 된다."""
    assert output_name("base64") == output_name("base64_float32")
    assert postprocess(POOLED, "base64") == postprocess(POOLED, "base64_float32")


def test_output_name_rejects_unknown():
    with pytest.raises(ValueError, match="int4"):
        output_name("int4")


def test_float_is_l2_normalized_tanh():
    out = postprocess(POOLED, "float")
    expected = np.tanh(POOLED)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    assert isinstance(out, list) and isinstance(out[0], list)
    assert np.allclose(np.array(out), expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(np.array(out), axis=-1), 1.0, atol=1e-6)


def test_base64_float32_is_little_endian():
    """확장 포맷 — float32 배열을 그대로 인코딩."""
    out = postprocess(POOLED, "base64_float32")
    decoded = np.frombuffer(base64.b64decode(out[0]), dtype="<f4")
    assert np.allclose(decoded, np.array(postprocess(POOLED, "float")[0]), atol=1e-6)


def test_base64_int8_round_trips_graph_output():
    """공식 출력(int8)을 손대지 않고 base64로만 감싼다."""
    graph_int8 = np.array([[85, -72, 0, 12, 5, -9, 30, -1]], dtype=np.int8)
    out = postprocess(graph_int8, "base64_int8")
    decoded = np.frombuffer(base64.b64decode(out[0]), dtype=np.int8)
    assert np.array_equal(decoded, graph_int8[0])


def test_base64_binary_packs_bits():
    """공식 PackedBinaryQuantizer: packbits(x >= 0). 8차원 → 1바이트."""
    graph_binary = np.where(POOLED >= 0, 1.0, -1.0).astype(np.float32)
    out = postprocess(graph_binary, "base64_binary")
    decoded = np.frombuffer(base64.b64decode(out[0]), dtype=np.uint8)
    assert len(decoded) == 1
    assert np.array_equal(decoded, np.packbits(POOLED[0] >= 0))


def test_postprocess_rejects_unknown():
    with pytest.raises(ValueError, match="int4"):
        postprocess(POOLED, "int4")


def test_zero_vector_does_not_produce_nan():
    out = postprocess(np.zeros((1, 8), dtype=np.float32), "float")
    assert not np.isnan(np.array(out)).any()


def test_truncate_none_is_identity():
    assert truncate(POOLED, None) is POOLED


def test_truncate_slices_leading_dimensions():
    """MRL은 앞쪽 차원을 남긴다."""
    out = truncate(POOLED, 4)
    assert out.shape == (2, 4)
    assert np.array_equal(out, POOLED[:, :4])


def test_truncated_float_is_renormalized():
    """자른 뒤 다시 L2 정규화돼야 한다 — 안 하면 길이가 1보다 작아진다."""
    out = np.array(postprocess(truncate(POOLED, 4), "float"))
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-6)


def test_truncation_commutes_with_quantization():
    """자르고 양자화한 것 == 양자화하고 자른 것.

    tanh/반올림/부호판정이 전부 elementwise라 성립한다. 이 성질이 깨지면
    그래프 출력을 잘라 쓰는 방식 자체가 틀린 것이 된다.
    """
    graph_int8 = np.round(np.tanh(POOLED) * 127).astype(np.int8)
    sliced_then = postprocess(truncate(graph_int8, 4), "base64_int8")
    then_sliced = [
        base64.b64encode(row[:4].tobytes()).decode() for row in graph_int8
    ]
    assert sliced_then == then_sliced


def test_all_schema_formats_are_mapped():
    """schemas.EncodingFormat과 OUTPUT_FOR가 어긋나지 않도록."""
    from typing import get_args

    from server.schemas import EncodingFormat

    assert set(get_args(EncodingFormat)) == set(OUTPUT_FOR) | set(ALIASES)
