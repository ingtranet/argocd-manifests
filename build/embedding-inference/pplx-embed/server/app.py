import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.concurrency import run_in_threadpool

from server.schemas import EmbeddingDatum, EmbeddingRequest, EmbeddingResponse, Usage

MODEL_ID = os.environ.get("MODEL_ID", "perplexity-ai/pplx-embed-v1-0.6b")

EMBED_LATENCY = Histogram(
    "pplx_embed_request_seconds", "End-to-end embedding request latency"
)
EMBED_TOTAL = Counter(
    "pplx_embed_embeddings_total", "Total embedded inputs", ["encoding_format"]
)
EMBED_ERRORS = Counter(
    "pplx_embed_errors_total", "Total failed embedding requests", ["reason"]
)

# Injection seam: tests set this directly; prod default is set at the bottom.
EMBEDDER_LOADER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if EMBEDDER_LOADER is None:
        raise RuntimeError("EMBEDDER_LOADER not set")
    app.state.embedder = EMBEDDER_LOADER()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest, request: Request):
    texts = [req.input] if isinstance(req.input, str) else req.input

    embedder = request.app.state.embedder
    try:
        with EMBED_LATENCY.time():
            vectors, tokens = await run_in_threadpool(
                embedder.embed, texts, req.encoding_format, req.dimensions
            )
        EMBED_TOTAL.labels(encoding_format=req.encoding_format).inc(len(texts))
    except Exception as e:
        EMBED_ERRORS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    return EmbeddingResponse(
        data=[EmbeddingDatum(index=i, embedding=v) for i, v in enumerate(vectors)],
        model=req.model or MODEL_ID,
        usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
    )


def _default_loader():
    from server.embedder import PplxEmbedder

    return PplxEmbedder.load()


# Production default. Tests override this before calling TestClient.
if EMBEDDER_LOADER is None:
    EMBEDDER_LOADER = _default_loader
