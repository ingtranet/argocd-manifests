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

        # Fuse Q/K/V projections in the transformer — single bigger matmul
        # instead of three, with a small RSS reduction (~weights of two
        # projections share a buffer) and a small speedup. No-op if the
        # transformer doesn't expose it.
        try:
            pipe.transformer.fuse_qkv_projections()
            print("[mem] transformer.fuse_qkv_projections() applied")
        except Exception as e:
            print(f"[mem] transformer.fuse_qkv_projections() skipped: {type(e).__name__}: {e}")
        # Sub-module slicing/tiling that the pipeline didn't expose at the
        # top level. VAE encode of multiple reference images is one of the
        # bigger transient peaks, so tile + slice it explicitly.
        for obj_name, fn_name in (("vae", "enable_slicing"), ("vae", "enable_tiling")):
            obj = getattr(pipe, obj_name, None)
            fn = getattr(obj, fn_name, None) if obj is not None else None
            if fn is None:
                print(f"[mem] pipe.{obj_name}.{fn_name}: not available, skip")
                continue
            try:
                fn()
                print(f"[mem] pipe.{obj_name}.{fn_name}() applied")
            except Exception as e:
                print(f"[mem] pipe.{obj_name}.{fn_name}() failed: {type(e).__name__}: {e}")

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
