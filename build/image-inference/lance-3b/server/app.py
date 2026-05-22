import base64
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from prometheus_client import Counter, Histogram, make_asgi_app

from server.schemas import ImageDatum, ImageRequest, ImageResponse

MODEL_ID = os.environ.get("MODEL_ID", "lance-3b-video-awq-int4")

GEN_LATENCY = Histogram(
    "lance_generation_seconds", "End-to-end image generation latency", ["task"]
)
GEN_TOTAL = Counter("lance_generations_total", "Total successful generations")
GEN_ERRORS = Counter(
    "lance_generation_errors_total", "Total failed generations", ["reason"]
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
            src_image = Image.open(
                io.BytesIO(base64.b64decode(req.image_b64))
            ).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"invalid image_b64: {e}")

    task = "image_edit" if src_image is not None else "t2i"
    try:
        with GEN_LATENCY.labels(task=task).time():
            if src_image is not None:
                images = await pipe.edit(
                    image=src_image, prompt=req.prompt, n=req.n,
                    seed=req.seed, width=width, height=height,
                )
            else:
                images = await pipe.generate(
                    prompt=req.prompt, n=req.n, seed=req.seed,
                    width=width, height=height,
                )
        GEN_TOTAL.inc(len(images))
    except Exception as e:
        GEN_ERRORS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    return ImageResponse(
        created=int(time.time()),
        data=[ImageDatum(b64_json=_png_b64(img)) for img in images],
    )


def _default_loader():
    from server.pipeline import LancePipeline

    return LancePipeline.load()


if PIPELINE_LOADER is None:
    PIPELINE_LOADER = _default_loader
