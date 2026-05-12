import base64
from io import BytesIO

from PIL import Image


def _decode_png(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


def test_generations_returns_b64_images(client, fake_pipeline):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "a hot air balloon", "n": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "created" in body
    assert len(body["data"]) == 2
    img = _decode_png(body["data"][0]["b64_json"])
    assert img.size == (512, 512)
    assert fake_pipeline.calls[0]["prompt"] == "a hot air balloon"
    assert fake_pipeline.calls[0]["n"] == 2


def test_generations_rejects_bad_size(client):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "x", "size": "384x384"},
    )
    assert r.status_code == 422  # Pydantic validation


def test_generations_rejects_url_format(client):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "x", "response_format": "url"},
    )
    assert r.status_code == 422


def test_generations_passes_seed(client, fake_pipeline):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "x", "seed": 12345},
    )
    assert r.status_code == 200
    assert fake_pipeline.calls[0]["seed"] == 12345


def test_generations_propagates_pipeline_error(client, fake_pipeline):
    async def boom(*, prompt, n, steps, guidance, seed):
        raise RuntimeError("OOM")

    fake_pipeline.generate = boom
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "x"},
    )
    assert r.status_code == 500
    assert "OOM" in r.json()["detail"]
