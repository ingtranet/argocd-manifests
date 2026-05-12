import asyncio
import os


class Flux2KleinPipeline:
    """Wraps diffusers Flux2KleinPipeline. Quantization is baked into the
    SDNQ ckpt — importing `sdnq` registers SDNQConfig with diffusers and
    transformers so from_pretrained transparently dequants on demand."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._lock = None

    @classmethod
    def load(cls) -> "Flux2KleinPipeline":
        import torch

        # Side-effect import: registers SDNQConfig globally.
        import sdnq  # noqa: F401
        from diffusers import Flux2KleinPipeline as DiffusersFlux2Klein

        model_id = os.environ.get(
            "FLUX_MODEL", "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic"
        )

        pipe = DiffusersFlux2Klein.from_pretrained(model_id, torch_dtype=torch.bfloat16)

        # apply_sdnq_options_to_model(use_quantized_matmul=True) requires
        # Triton, which is not available on Jetson. SDNQ still dequants
        # weights transparently — we just lose the optional INT8 matmul fast
        # path and fall back to PyTorch eager bf16 matmul.
        pipe.to("cuda")
        return cls(pipe)

    async def generate(self, *, prompt, n, steps, guidance, seed, width, height):
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._run, prompt, n, steps, guidance, seed, width, height
            )

    def _run(self, prompt, n, steps, guidance, seed, width, height):
        import torch

        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )
        out = self._pipe(
            prompt=[prompt] * n,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        return out.images
