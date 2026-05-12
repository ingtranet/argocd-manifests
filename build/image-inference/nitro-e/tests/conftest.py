import pytest
from PIL import Image


class FakePipeline:
    """Mock NitroEPipeline that returns dummy PIL images without a GPU."""

    def __init__(self, model_id="amd/Nitro-E-dist"):
        self.model_id = model_id
        self.calls = []

    async def generate(self, *, prompt, n, steps, guidance, seed):
        self.calls.append(
            {"prompt": prompt, "n": n, "steps": steps, "guidance": guidance, "seed": seed}
        )
        return [Image.new("RGB", (512, 512), color=(10, 20, 30)) for _ in range(n)]


@pytest.fixture
def fake_pipeline():
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline, monkeypatch):
    # Defer imports until after monkeypatching so app.lifespan sees fake.
    monkeypatch.setenv("MODEL_ID", "amd/Nitro-E-dist")
    from server import app as app_module

    app_module.PIPELINE_LOADER = lambda: fake_pipeline  # injection seam
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
