import asyncio
import os


class Flux2Klein9BPipeline:
    """Wraps diffusers Flux2KleinPipeline. Loads a *pre-quantized* bnb NF4
    checkpoint from disk — no on-the-fly quantization at startup, because
    that path peaks at ~13GB+ on 9B and OOMs on the Orin NX's 16GB unified
    memory. The prequantized weights (transformer + Qwen3 text encoder)
    mmap-in at their NF4 footprint (~5GB + ~6GB).

    Inference still uses the Jetson-built bitsandbytes in dustynv/vllm
    (libbitsandbytes_cuda129.so with sm_87 cubin) for NF4 dequant. Pipeline
    is kept fully resident via pipe.to('cuda'); CPU offload is pointless on
    Orin (cuda/cpu share one DRAM pool).

    Single-GPU serialization via lazy asyncio.Lock. One instance per process.
    """

    def __init__(self, pipe):
        self._pipe = pipe
        self._lock = None

    @classmethod
    def load(cls) -> "Flux2Klein9BPipeline":
        import torch
        from diffusers import Flux2KleinPipeline as DiffusersFlux2Klein

        model_id = os.environ.get(
            "FLUX_MODEL", "cookieshake/FLUX.2-klein-9B-bnb-nf4"
        )
        pipe = DiffusersFlux2Klein.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        # Memory-saving knobs for the Orin NX 16GB unified-memory case.
        # Single-image generate/edit fits comfortably, but multi-reference
        # edit (>=2 images) blows the peak: ref VAE encode + concatenated
        # ref+noisy latents push transformer attention activations past
        # the 3-4GB headroom we have over the 11GB weights.
        # These knobs are no-ops if Flux2KleinPipeline doesn't expose them.
        knobs = (
            ("enable_attention_slicing", ("max",)),
            ("enable_vae_slicing", ()),
            ("enable_vae_tiling", ()),
        )
        for name, args in knobs:
            fn = getattr(pipe, name, None)
            if fn is None:
                print(f"[mem] {name}: pipeline does not expose it, skip")
                continue
            try:
                fn(*args)
                print(f"[mem] {name}{args} applied")
            except Exception as e:
                print(f"[mem] {name}{args} failed: {type(e).__name__}: {e}")

        return cls(pipe)

    async def generate(self, *, prompt, n, steps, guidance, seed, width, height):
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._run, prompt, n, steps, guidance, seed, width, height, None
            )

    async def edit(self, *, image, prompt, n, steps, guidance, seed, width, height):
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._run, prompt, n, steps, guidance, seed, width, height, image
            )

    def _run(self, prompt, n, steps, guidance, seed, width, height, image):
        import torch

        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )
        call_kwargs = dict(
            prompt=[prompt] * n,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if image is not None:
            call_kwargs["image"] = image
        out = self._pipe(**call_kwargs)
        return out.images
