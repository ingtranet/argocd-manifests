import base64
import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from server.schemas import ImageDatum, ImageRequest, ImageResponse

MODEL_ID = os.environ.get("MODEL_ID", "amd/Nitro-E-dist")
DEFAULT_STEPS = int(os.environ.get("NITRO_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("NITRO_GUIDANCE", "0.0"))

# Injection seam: tests set this directly; prod main() sets the real loader.
PIPELINE_LOADER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PIPELINE_LOADER is None:
        raise RuntimeError("PIPELINE_LOADER not set")
    loader = PIPELINE_LOADER
    pipe = loader()
    app.state.pipe = pipe
    yield


app = FastAPI(lifespan=lifespan)


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
    try:
        images = await pipe.generate(
            prompt=req.prompt,
            n=req.n,
            steps=DEFAULT_STEPS,
            guidance=DEFAULT_GUIDANCE,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ImageResponse(
        created=int(time.time()),
        data=[ImageDatum(b64_json=_png_b64(img)) for img in images],
    )
