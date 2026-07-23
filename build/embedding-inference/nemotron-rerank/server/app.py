import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.concurrency import run_in_threadpool

from server.schemas import RerankRequest, RerankResponse, RerankResult

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/llama-nemotron-rerank-1b-v2")

RERANK_LATENCY = Histogram(
    "nemotron_rerank_request_seconds", "End-to-end rerank request latency"
)
RERANK_TOTAL = Counter(
    "nemotron_rerank_total", "Total reranked document pairs"
)
RERANK_ERRORS = Counter(
    "nemotron_rerank_errors_total", "Total failed rerank requests", ["reason"]
)

# Injection seam: tests set this directly; prod default is set at the bottom.
RERANKER_LOADER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RERANKER_LOADER is None:
        raise RuntimeError("RERANKER_LOADER not set")
    app.state.reranker = RERANKER_LOADER()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest, request: Request):
    reranker = request.app.state.reranker
    try:
        with RERANK_LATENCY.time():
            scored = await run_in_threadpool(
                reranker.rerank, req.query, req.documents, req.top_n
            )
        RERANK_TOTAL.inc(len(req.documents))
    except Exception as e:
        RERANK_ERRORS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    return RerankResponse(
        model=req.model or MODEL_ID,
        results=[
            RerankResult(index=idx, relevance_score=score)
            for idx, score in scored
        ],
    )


def _default_loader():
    from server.reranker import NemotronReranker

    return NemotronReranker.load()


# Production default. Tests override this before calling TestClient.
if RERANKER_LOADER is None:
    RERANKER_LOADER = _default_loader
