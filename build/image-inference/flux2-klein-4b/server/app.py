import base64
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

from server.schemas import ImageDatum, ImageRequest, ImageResponse

MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_STEPS = int(os.environ.get("FLUX_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("FLUX_GUIDANCE", "0.0"))

GEN_LATENCY = Histogram(
    "flux2_klein_generation_seconds", "End-to-end image generation latency"
)
GEN_TOTAL = Counter("flux2_klein_generations_total", "Total successful generations")
GEN_ERRORS = Counter(
    "flux2_klein_generation_errors_total",
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
    try:
        with GEN_LATENCY.time():
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


def _default_loader():
    from server.pipeline import Flux2KleinPipeline

    return Flux2KleinPipeline.load()


if PIPELINE_LOADER is None:
    PIPELINE_LOADER = _default_loader
