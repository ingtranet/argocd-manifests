import asyncio
import os


class Flux2Klein9BPipeline:
    """Wraps diffusers Flux2KleinPipeline.

    Base is a *pre-quantized* bnb NF4 checkpoint loaded from disk. LoRAs
    listed in LORA_REPOS env are loaded via PEFT then their lora_A/lora_B
    nn.Linear modules are replaced in-place by bnb.nn.Linear4bit so the
    adapter weights themselves live in NF4 (≈4x smaller than bf16). This
    keeps the option to swap adapter scales at config time (unlike baking
    into the base) while cutting the resident adapter footprint on the
    16 GiB unified-memory Orin NX.

    Knobs:
      FLUX_MODEL         base repo id (default pre-quant NF4)
      LORA_REPOS         comma list: repo[:scale][:adapter_name], ...
                         e.g. dx8152/Flux2-Klein-9B-Enhanced-Details:1.0:enh,
                              dx8152/Flux2-Klein-9B-Consistency:0.8:cons
      LORA_QUANT         nf4 | int8 | off   (default nf4)
    """

    def __init__(self, pipe):
        self._pipe = pipe
        self._lock = None

    @classmethod
    def load(cls) -> "Flux2Klein9BPipeline":
        import torch
        from diffusers import Flux2KleinPipeline as DiffusersFlux2Klein

        model_id = os.environ.get("FLUX_MODEL", "cookieshake/FLUX.2-klein-9B-bnb-nf4")
        pipe = DiffusersFlux2Klein.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.to("cuda")

        cls._load_loras_quantized(pipe)

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

    @staticmethod
    def _parse_lora_spec(spec: str):
        # repo[:scale[:adapter_name[:weight_name]]]
        parts = spec.strip().split(":")
        repo = parts[0]
        scale = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
        name = parts[2] if len(parts) > 2 and parts[2] else repo.split("/")[-1].replace("-", "_").lower()
        weight_name = parts[3] if len(parts) > 3 and parts[3] else None
        return repo, scale, name, weight_name

    @classmethod
    def _load_loras_quantized(cls, pipe):
        spec_env = os.environ.get("LORA_REPOS", "").strip()
        if not spec_env:
            print("[lora] LORA_REPOS empty; no adapters loaded")
            return

        specs = [cls._parse_lora_spec(s) for s in spec_env.split(",") if s.strip()]
        names, scales = [], []
        for repo, scale, name, weight_name in specs:
            print(f"[lora] loading {repo} (file={weight_name}) as '{name}' scale={scale}")
            kwargs = {"adapter_name": name}
            if weight_name:
                kwargs["weight_name"] = weight_name
            pipe.load_lora_weights(repo, **kwargs)
            names.append(name)
            scales.append(scale)
        try:
            pipe.set_adapters(names, adapter_weights=scales)
            print(f"[lora] active adapters: {names} scales={scales}")
        except Exception as e:
            print(f"[lora] set_adapters failed: {type(e).__name__}: {e}")

        quant = os.environ.get("LORA_QUANT", "nf4").lower()
        if quant == "off":
            print("[lora] LORA_QUANT=off; skipping in-place quantization")
            return
        cls._quantize_lora_modules(pipe.transformer, quant)

    @staticmethod
    def _quantize_lora_modules(root, quant: str):
        import torch
        import bitsandbytes as bnb
        from peft.tuners.lora import LoraLayer

        replaced = 0
        skipped = 0
        bytes_before = 0
        bytes_after_est = 0

        def make_4bit(lin: torch.nn.Linear) -> torch.nn.Module:
            nonlocal bytes_before, bytes_after_est
            w = lin.weight.data
            bytes_before += w.numel() * w.element_size()
            new_lin = bnb.nn.Linear4bit(
                lin.in_features,
                lin.out_features,
                bias=lin.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )
            new_lin.weight = bnb.nn.Params4bit(
                w.detach().to("cpu").contiguous(),
                requires_grad=False,
                quant_type="nf4",
            )
            if lin.bias is not None:
                new_lin.bias = torch.nn.Parameter(lin.bias.data.detach().clone(), requires_grad=False)
            new_lin = new_lin.to(w.device)
            bytes_after_est += w.numel() // 2 + 32  # nf4 ~4-bit weight + small overhead per block
            return new_lin

        def make_8bit(lin: torch.nn.Linear) -> torch.nn.Module:
            nonlocal bytes_before, bytes_after_est
            w = lin.weight.data
            bytes_before += w.numel() * w.element_size()
            new_lin = bnb.nn.Linear8bitLt(
                lin.in_features,
                lin.out_features,
                bias=lin.bias is not None,
                has_fp16_weights=False,
            )
            new_lin.weight = bnb.nn.Int8Params(
                w.detach().to("cpu").contiguous(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if lin.bias is not None:
                new_lin.bias = torch.nn.Parameter(lin.bias.data.detach().clone(), requires_grad=False)
            new_lin = new_lin.to(w.device)
            bytes_after_est += w.numel()
            return new_lin

        make = make_4bit if quant == "nf4" else make_8bit

        for mod in root.modules():
            if not isinstance(mod, LoraLayer):
                continue
            for bag_name in ("lora_A", "lora_B"):
                bag = getattr(mod, bag_name, None)
                if bag is None:
                    continue
                for adapter_name in list(bag.keys()):
                    lin = bag[adapter_name]
                    if not isinstance(lin, torch.nn.Linear):
                        skipped += 1
                        continue
                    try:
                        bag[adapter_name] = make(lin)
                        replaced += 1
                    except Exception as e:
                        skipped += 1
                        print(f"[lora-quant] {bag_name}.{adapter_name} replace failed: {type(e).__name__}: {e}")

        print(
            f"[lora-quant] quant={quant} replaced={replaced} skipped={skipped} "
            f"bf16_bytes={bytes_before/1e6:.1f}MB -> est={bytes_after_est/1e6:.1f}MB"
        )

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
