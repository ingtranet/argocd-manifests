import asyncio
import os


class Flux2Klein9BPipeline:
    """Wraps diffusers Flux2KleinPipeline. Loads a *pre-quantized* bnb NF4
    checkpoint from disk with dx8152's Enhanced-Details (1.0) and
    Consistency-V2 (0.8) LoRAs already baked into the transformer weights.

    LoRAs are not loaded at runtime — they were merged into the bf16
    transformer once on Mac CPU and the result re-quantized to NF4. This
    means: no PEFT hooks, no extra forward path, no transient adapter
    delta-weight buffers. Inference cost matches a vanilla NF4 base.

    Inference uses the Jetson-built bitsandbytes in dustynv/vllm
    (libbitsandbytes_cuda129.so with sm_87 cubin) for NF4 dequant. Pipeline
    is kept fully resident via pipe.to('cuda'); CPU offload is pointless on
    Orin (cuda/cpu share one DRAM pool). Attention slicing is left on for
    safety on multi-reference edit, where peak activation memory still
    grows ~O(seq^2).

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
            "FLUX_MODEL", "cookieshake/FLUX.2-klein-9B-bnb-nf4-loras-baked"
        )
        pipe = DiffusersFlux2Klein.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        # Memory-saving knobs for the Orin NX 16GB unified-memory case.
        # Best-effort: a missing API on the Flux2 pipeline is a no-op.
        for obj_name, fn_name in (("vae", "enable_slicing"), ("vae", "enable_tiling")):
            obj = getattr(pipe, obj_name, None)
            fn = getattr(obj, fn_name, None) if obj is not None else None
            if fn is not None:
                try:
                    fn()
                    print(f"[mem] pipe.{obj_name}.{fn_name}() applied")
                except Exception as e:
                    print(f"[mem] pipe.{obj_name}.{fn_name}() failed: {type(e).__name__}: {e}")
        try:
            pipe.enable_attention_slicing("max")
            print("[mem] enable_attention_slicing('max') applied")
        except Exception as e:
            print(f"[mem] enable_attention_slicing skipped: {type(e).__name__}: {e}")

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
