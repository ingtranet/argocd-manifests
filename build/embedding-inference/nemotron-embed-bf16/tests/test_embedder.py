"""Unit tests for the embedder module (no model required)."""
import numpy as np
import pytest

from server.embedder import postprocess


def test_postprocess_float_returns_list():
    embeddings = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = postprocess(embeddings, "float")
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert all(isinstance(v, float) for v in result[0])


def test_postprocess_base64_returns_strings():
    embeddings = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    result = postprocess(embeddings, "base64")
    assert isinstance(result, list)
    assert isinstance(result[0], str)


def test_postprocess_invalid_format_raises():
    embeddings = np.array([[0.5, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid encoding_format"):
        postprocess(embeddings, "base64_int8")
