import asyncio
import os


class Flux2KleinPipeline:
    """Wraps diffusers Flux2KleinPipeline with on-the-fly bitsandbytes NF4
    quantization on the transformer and the Qwen3 text encoder. Single-GPU
    serialization via lazy asyncio.Lock. One instance per process."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._lock = None  # created lazily inside the running event loop

    @classmethod
    def load(cls) -> "Flux2KleinPipeline":
        import torch
        from diffusers import (
            Flux2KleinPipeline as DiffusersFlux2Klein,
            Flux2Transformer2DModel,
            BitsAndBytesConfig as DiffusersBnbConfig,
        )
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig as TransformersBnbConfig,
        )

        model_id = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
        nf4 = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        transformer = Flux2Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=DiffusersBnbConfig(**nf4),
            torch_dtype=torch.bfloat16,
        )
        text_encoder = AutoModelForCausalLM.from_pretrained(
            model_id,
            subfolder="text_encoder",
            quantization_config=TransformersBnbConfig(**nf4),
            torch_dtype=torch.bfloat16,
        )
        pipe = DiffusersFlux2Klein.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
        )
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
