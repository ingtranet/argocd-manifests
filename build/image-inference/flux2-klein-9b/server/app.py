import base64
import io
import os
import time
from contextlib import asynccontextmanager

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app

from server.schemas import (
    ALLOWED_SIZES,
    ImageDatum,
    ImageRequest,
    ImageResponse,
)

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
DEFAULT_STEPS = int(os.environ.get("FLUX_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "0.0"))

GEN_LATENCY = Histogram(
    "flux2_klein_9b_generation_seconds", "End-to-end image generation latency"
)
GEN_TOTAL = Counter("flux2_klein_9b_generations_total", "Total successful generations")
GEN_ERRORS = Counter(
    "flux2_klein_9b_generation_errors_total",
    "Total failed generations",
    ["reason"],
)

PIPELINE_LOADER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PIPELINE_LOADER is None:
        raise RuntimeError("PIPELINE_LOADER not set")
    app.state.pipe = PIPELINE_LOADER()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/v1/images/generations", response_model=ImageResponse)
async def generations(req: ImageRequest, request: Request):
    pipe = request.app.state.pipe
    width, height = (int(x) for x in req.size.split("x"))

    src_image = None
    if req.image_b64:
        try:
            src_image = Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"invalid image_b64: {e}")

    try:
        with GEN_LATENCY.time():
            if src_image is not None:
                images = await pipe.edit(
                    image=src_image,
                    prompt=req.prompt,
                    n=req.n,
                    steps=DEFAULT_STEPS,
                    guidance=DEFAULT_GUIDANCE,
                    seed=req.seed,
                    width=width,
                    height=height,
                )
            else:
                images = await pipe.generate(
                    prompt=req.prompt,
                    n=req.n,
                    steps=DEFAULT_STEPS,
                    guidance=DEFAULT_GUIDANCE,
                    seed=req.seed,
                    width=width,
                    height=height,
                )
        GEN_TOTAL.inc(req.n)
    except Exception as e:
        GEN_ERRORS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    return ImageResponse(
        created=int(time.time()),
        data=[ImageDatum(b64_json=_png_b64(img)) for img in images],
    )


@app.post("/v1/images/edits", response_model=ImageResponse)
async def edits(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    n: int = Form(1),
    size: str = Form("512x512"),
    response_format: str = Form("b64_json"),
    seed: Optional[int] = Form(None),
):
    if size not in ALLOWED_SIZES:
        raise HTTPException(status_code=422, detail=f"size must be one of {sorted(ALLOWED_SIZES)}")
    if response_format != "b64_json":
        raise HTTPException(status_code=422, detail="response_format must be b64_json")
    if not (1 <= n <= 4):
        raise HTTPException(status_code=422, detail="n must be between 1 and 4")
    if not prompt or len(prompt) > 2000:
        raise HTTPException(status_code=422, detail="prompt missing or too long")

    width, height = (int(x) for x in size.split("x"))
    img_bytes = await image.read()
    try:
        src = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"invalid image: {e}")

    pipe = request.app.state.pipe
    try:
        with GEN_LATENCY.time():
            images = await pipe.edit(
                image=src,
                prompt=prompt,
                n=n,
                steps=DEFAULT_STEPS,
                guidance=DEFAULT_GUIDANCE,
                seed=seed,
                width=width,
                height=height,
            )
        GEN_TOTAL.inc(n)
    except Exception as e:
        GEN_ERRORS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    return ImageResponse(
        created=int(time.time()),
        data=[ImageDatum(b64_json=_png_b64(out)) for out in images],
    )


def _default_loader():
    from server.pipeline import Flux2Klein9BPipeline

    return Flux2Klein9BPipeline.load()


if PIPELINE_LOADER is None:
    PIPELINE_LOADER = _default_loader
