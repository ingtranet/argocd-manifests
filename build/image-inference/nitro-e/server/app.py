import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

MODEL_ID = os.environ.get("MODEL_ID", "amd/Nitro-E-dist")

# Injection seam: overridden by tests; in prod set by Task 6 to a real loader.
PIPELINE_LOADER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PIPELINE_LOADER is None:
        raise RuntimeError("PIPELINE_LOADER not set")
    app.state.pipe = PIPELINE_LOADER()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}
