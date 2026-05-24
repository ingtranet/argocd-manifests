"""Serving wrapper around lance-quant's persistent `lance_worker.py`.

Lance is a custom (non-HF) multimodal model and AutoAWQ/vLLM cannot serve it.
The AWQ author ships a persistent worker that loads Lance once and answers
line-delimited JSON requests over stdin/stdout. We reuse that worker verbatim
(vendored at /opt/lance-quant) rather than reimplementing the fragile build +
generation glue, and put a FastAPI shell in front of it that mirrors the
flux2-klein OpenAI-style image API.

One worker == one GPU == strictly serialized requests (asyncio.Lock).

Caveats baked into this backend (see the AWQ model card):
  * pure-PyTorch INT4 dequant runs ~10x slower than bf16 — expect minutes per
    image on Jetson Orin. This PoC exists to measure that in practice.
  * the v1 AWQ calibration only covered t2i + x2t_image, so image_edit quality
    may be degraded relative to the bf16 baseline.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

LANCE_SRC = os.environ.get("LANCE_SRC", "/opt/Lance")
LANCE_QUANT = os.environ.get("LANCE_QUANT", "/opt/lance-quant")
# Path to the original Lance checkpoint dir (config/tokenizer/ViT/VAE come from
# here; the AWQ dir only overrides the LLM linears).
MODEL_PATH = os.environ.get("LANCE_MODEL_PATH", "/models/Lance/Lance_3B_Video")
VIT_PATH = os.environ.get("LANCE_VIT_PATH", "/models/Lance/Qwen2.5-VL-ViT")
# AWQ INT4 checkpoint dir (awq_state_dict.safetensors + awq_meta.json). Leave
# empty to run the bf16 baseline instead.
AWQ_DIR = os.environ.get("LANCE_AWQ_DIR", "/models/Lance-3B-Video-AWQ-INT4")

WORKER_READY_TIMEOUT = int(os.environ.get("LANCE_READY_TIMEOUT", "1800"))

# size string -> (width, height)
_SIZE_MAP = {"768x768": (768, 768), "512x512": (512, 512)}


class LanceWorkerError(RuntimeError):
    pass


class LancePipeline:
    """Owns the long-lived lance_worker subprocess and serializes access."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._lock = None

    @classmethod
    def load(cls) -> "LancePipeline":
        worker = os.path.join(LANCE_QUANT, "comfyui", "lance_worker.py")
        argv = [
            sys.executable, worker,
            "--lance_src", LANCE_SRC,
            "--script_root", LANCE_QUANT,
            "--model_path", MODEL_PATH,
            "--vit_path", VIT_PATH,
            "--save_path_gen", "/tmp/lance_bootstrap",
        ]
        if AWQ_DIR:
            argv += ["--awq_dir", AWQ_DIR]

        # cwd must be the Lance source so its relative config paths
        # (config/examples/*.json, config_factory) resolve.
        proc = subprocess.Popen(
            argv,
            cwd=LANCE_SRC,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let build logs flow to the container's stderr
            text=True,
            bufsize=1,
        )

        # Worker prints build progress to stderr and a single "READY" line to
        # stdout once the model is loaded + bootstrapped.
        deadline = time.time() + WORKER_READY_TIMEOUT
        while True:
            if proc.poll() is not None:
                raise LanceWorkerError(
                    f"lance_worker exited during startup (code {proc.returncode})"
                )
            if time.time() > deadline:
                proc.kill()
                raise LanceWorkerError("lance_worker did not become READY in time")
            line = proc.stdout.readline()
            if not line:
                continue
            if line.strip() == "READY":
                break
        return cls(proc)

    async def generate(self, *, prompt, n, seed, width, height):
        return await self._run("t2i", prompt, None, n, seed, width, height)

    async def edit(self, *, image, prompt, n, seed, width, height):
        return await self._run("image_edit", prompt, image, n, seed, width, height)

    async def _run(self, task, prompt, image, n, seed, width, height):
        if self._lock is None:
            self._lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                None, self._run_sync, task, prompt, image, n, seed, width, height
            )

    def _run_sync(self, task, prompt, image, n, seed, width, height):
        if self._proc.poll() is not None:
            raise LanceWorkerError(
                f"lance_worker is not running (code {self._proc.returncode})"
            )

        work = Path(tempfile.mkdtemp(prefix="lance_"))
        save_dir = work / "out"
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = work / "manifest.json"

        if task == "t2i":
            # filename -> prompt; one entry per requested image
            manifest = {f"out-{i:02d}.png": prompt for i in range(n)}
        else:  # image_edit
            src = work / "input.png"
            image.save(src, format="PNG")
            manifest = {
                f"edit-{i:02d}": {
                    "interleave_array": [prompt, str(src), str(src)],
                    "element_dtype_array": ["text", "image", "image"],
                    "istarget_in_interleave": [0, 0, 1],
                }
                for i in range(n)
            }
        manifest_path.write_text(json.dumps(manifest))

        req = {
            "task": task,
            "manifest_path": str(manifest_path),
            "save_dir": str(save_dir),
            "width": width,
            "height": height,
        }
        if seed is not None:
            req["seed"] = seed

        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()

        resp_line = self._proc.stdout.readline()
        if not resp_line:
            raise LanceWorkerError("lance_worker closed stdout without responding")
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise LanceWorkerError(resp.get("error", "unknown worker error"))

        images = [
            Image.open(p).convert("RGB")
            for p in sorted(save_dir.glob("*.png")) + sorted(save_dir.glob("*.jpg"))
        ]
        if not images:
            raise LanceWorkerError("worker reported ok but produced no images")
        return images
