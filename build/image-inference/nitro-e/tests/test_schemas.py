import pytest
from pydantic import ValidationError

from server.schemas import ImageRequest


def test_minimal_valid_request():
    req = ImageRequest(prompt="a cat")
    assert req.prompt == "a cat"
    assert req.n == 1
    assert req.size == "512x512"
    assert req.response_format == "b64_json"
    assert req.seed is None


def test_full_valid_request():
    req = ImageRequest(
        prompt="a cat",
        model="amd/Nitro-E-dist",
        n=2,
        size="512x512",
        response_format="b64_json",
        seed=42,
    )
    assert req.n == 2
    assert req.seed == 42


def test_n_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", n=0)
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", n=5)


def test_unsupported_size_rejected():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", size="384x384")


def test_url_response_format_rejected():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", response_format="url")


def test_prompt_length_capped():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x" * 2001)
