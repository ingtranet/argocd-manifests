import asyncio
import sys
import types

import pytest


@pytest.fixture
def stub_init_pipe(monkeypatch):
    """Stub out core.tools.inference_pipe.init_pipe + torch for unit tests."""
    # stub torch
    fake_torch = types.SimpleNamespace(
        device=lambda x: x,
        bfloat16="bfloat16",
        Generator=lambda device: types.SimpleNamespace(
            manual_seed=lambda s: types.SimpleNamespace(_seed=s)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # stub core.tools.inference_pipe
    calls = {}

    class FakePipe:
        def __call__(self, *, prompt, width, height, num_inference_steps, guidance_scale, generator=None):
            calls["last"] = {
                "prompt": prompt, "width": width, "height": height,
                "steps": num_inference_steps, "guidance": guidance_scale,
                "generator": generator,
            }
            from PIL import Image
            n = len(prompt) if isinstance(prompt, list) else 1
            return types.SimpleNamespace(
                images=[Image.new("RGB", (width, height)) for _ in range(n)]
            )

    def init_pipe(device, dtype, resolution, *, repo_name, ckpt_name, ckpt_path_grpo=None):
        calls["init"] = {
            "device": device, "dtype": dtype, "resolution": resolution,
            "repo_name": repo_name, "ckpt_name": ckpt_name,
        }
        return FakePipe()

    module = types.ModuleType("core.tools.inference_pipe")
    module.init_pipe = init_pipe
    monkeypatch.setitem(sys.modules, "core", types.ModuleType("core"))
    monkeypatch.setitem(sys.modules, "core.tools", types.ModuleType("core.tools"))
    monkeypatch.setitem(sys.modules, "core.tools.inference_pipe", module)

    return calls


def test_load_calls_init_pipe_with_dist_checkpoint(stub_init_pipe, monkeypatch):
    monkeypatch.setenv("NITRO_CKPT", "Nitro-E-512px-dist.safetensors")
    from server.pipeline import NitroEPipeline
    NitroEPipeline.load()
    assert stub_init_pipe["init"]["ckpt_name"] == "Nitro-E-512px-dist.safetensors"
    assert stub_init_pipe["init"]["resolution"] == 512


def test_generate_passes_prompt_n_times(stub_init_pipe):
    from server.pipeline import NitroEPipeline
    p = NitroEPipeline.load()
    images = asyncio.run(p.generate(prompt="cat", n=3, steps=4, guidance=0.0, seed=None))
    assert len(images) == 3
    assert stub_init_pipe["last"]["prompt"] == ["cat", "cat", "cat"]
    assert stub_init_pipe["last"]["steps"] == 4
    assert stub_init_pipe["last"]["guidance"] == 0.0


def test_generate_uses_seed_when_given(stub_init_pipe):
    from server.pipeline import NitroEPipeline
    p = NitroEPipeline.load()
    asyncio.run(p.generate(prompt="cat", n=1, steps=4, guidance=0.0, seed=42))
    gen = stub_init_pipe["last"]["generator"]
    assert getattr(gen, "_seed", None) == 42
