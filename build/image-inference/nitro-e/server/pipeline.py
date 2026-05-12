import asyncio
import os


class NitroEPipeline:
    """Wraps AMD-AGI/Nitro-E init_pipe with an asyncio.Lock for single-GPU
    serialization. One instance per process."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._lock = None  # created lazily inside the running event loop

    @classmethod
    def load(cls) -> "NitroEPipeline":
        # Lazy imports so each test's sys.modules monkeypatch is honored.
        # In production these are sys.modules cache hits after the first call.
        import torch
        from core.tools.inference_pipe import init_pipe

        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        repo_name = os.environ.get("NITRO_REPO", "amd/Nitro-E")
        ckpt_name = os.environ.get("NITRO_CKPT", "Nitro-E-512px-dist.safetensors")
        pipe = init_pipe(
            device,
            dtype,
            512,
            repo_name=repo_name,
            ckpt_name=ckpt_name,
        )
        return cls(pipe)

    async def generate(self, *, prompt, n, steps, guidance, seed):
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._run, prompt, n, steps, guidance, seed
            )

    def _run(self, prompt, n, steps, guidance, seed):
        import torch

        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )
        out = self._pipe(
            prompt=[prompt] * n,
            width=512,
            height=512,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return out.images
