#!/usr/bin/env python3
"""
Jina Reranker v3 HTTP wrapper for llama.cpp on Jetson Orin.

Implements POST /v1/rerank (standard OpenAI-compatible rerank schema),
GET  /health, and GET /metrics.

Uses llama-embedding CLI with --pooling none to extract per-token hidden
states, applies the MLP projector from projector.safetensors, computes
cosine similarity between query and document embeddings, and returns
sorted relevance scores.

Reference: https://huggingface.co/jinaai/jina-reranker-v3-GGUF
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jina-reranker")

# ---------------------------------------------------------------------------
# Configuration (env overrides)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/jinaai_jina-reranker-v3-GGUF/jina-reranker-v3-Q8_0.gguf")
PROJECTOR_PATH = os.environ.get("PROJECTOR_PATH", "/models/jinaai_jina-reranker-v3-GGUF/projector.safetensors")
LLAMA_EMBEDDING_PATH = os.environ.get("LLAMA_EMBEDDING_PATH", "/usr/local/bin/llama-embedding")
LLAMA_TOKENIZE_PATH = os.environ.get("LLAMA_TOKENIZE_PATH", "/usr/local/bin/llama-tokenize")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
NGL = int(os.environ.get("NGL", "99"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
UBATCH_SIZE = int(os.environ.get("UBATCH_SIZE", "512"))
HF_REPO = os.environ.get("HF_REPO", "jinaai/jina-reranker-v3-GGUF")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Special tokens (Jina v3)
# ---------------------------------------------------------------------------
QUERY_EMBED_TOKEN = "<|rerank_token|>"
DOC_EMBED_TOKEN = "<|embed_token|>"
QUERY_EMBED_TOKEN_ID = 151671
DOC_EMBED_TOKEN_ID = 151670
EMBD_SEPARATOR = "<#JINA_SEP#>"

# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------
class MLPProjector:
    """MLP projector: hidden states -> embedding space (2-layer MLP with ReLU)."""

    def __init__(self, linear1_weight: np.ndarray, linear2_weight: np.ndarray):
        self.linear1_weight = linear1_weight  # [hidden, 512]
        self.linear2_weight = linear2_weight  # [512, 512]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x @ self.linear1_weight.T
        x = np.maximum(0, x)  # ReLU
        x = x @ self.linear2_weight.T
        return x


def load_projector(path: str) -> MLPProjector:
    """Load projector weights from safetensors file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Projector not found: {path}")
    with safe_open(path, framework="numpy") as f:
        w0 = f.get_tensor("projector.0.weight")
        w2 = f.get_tensor("projector.2.weight")
    log.info("Projector loaded: linear1 %s, linear2 %s", w0.shape, w2.shape)
    return MLPProjector(w0, w2)


# ---------------------------------------------------------------------------
# Prompt formatting (Jina v3)
# ---------------------------------------------------------------------------
def sanitize_input(text: str) -> str:
    """Remove special tokens from input text."""
    for token in (QUERY_EMBED_TOKEN, DOC_EMBED_TOKEN):
        text = text.replace(token, "")
    return text


def format_rerank_prompt(
    query: str,
    documents: List[str],
    instruction: Optional[str] = None,
) -> str:
    """Format query and documents into Jina v3 rerank prompt."""
    query = sanitize_input(query)
    documents = [sanitize_input(d) for d in documents]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages "
        "based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it "
        "answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage "
        "satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining "
        "the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"

    prompt = (
        f"I will provide you with {len(documents)} passages, each indicated by a "
        f"numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f"<instruct>\n{instruction}\n</instruct>\n"

    doc_prompts = [
        f'<passage id="{i}">\n{doc}{DOC_EMBED_TOKEN}\n</passage>'
        for i, doc in enumerate(documents)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{QUERY_EMBED_TOKEN}\n</query>"

    return prefix + prompt + suffix


# ---------------------------------------------------------------------------
# llama-embedding CLI wrapper
# ---------------------------------------------------------------------------
async def run_llama_embedding(prompt: str) -> np.ndarray:
    """Run llama-embedding and return per-token hidden states as [n_tokens, hidden_dim]."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", prefix="jina_prompt_"
    ) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            LLAMA_EMBEDDING_PATH,
            "-m", MODEL_PATH,
            "-f", prompt_file,
            "--pooling", "none",
            "--embd-separator", EMBD_SEPARATOR,
            "--embd-normalize", "-1",
            "--embd-output-format", "json",
            "--ubatch-size", str(UBATCH_SIZE),
            "--ctx-size", str(CTX_SIZE),
            "--flash-attn", "auto",
            "-ngl", str(NGL),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-embedding exited {proc.returncode}: {stderr_text[:2000]}"
            )

        output = json.loads(stdout.decode("utf-8"))
        embeddings = [item["embedding"] for item in output["data"]]
        return np.array(embeddings, dtype=np.float32)
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


async def run_llama_tokenize(prompt: str) -> List[int]:
    """Tokenize prompt to find special token positions."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", prefix="jina_tokenize_"
    ) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            LLAMA_TOKENIZE_PATH,
            "-m", MODEL_PATH,
            "-f", prompt_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError("llama-tokenize failed")

        tokens = []
        for line in stdout.decode("utf-8").strip().split("\n"):
            if "->" in line:
                token_id = int(line.split("->")[0].strip())
                tokens.append(token_id)
        return tokens
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Rerank logic
# ---------------------------------------------------------------------------
def compute_scores(
    query_embed: np.ndarray, doc_embeds: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between query and each document embedding."""
    # query_embed: [1, 512], doc_embeds: [n_docs, 512]
    dot = np.sum(doc_embeds * query_embed, axis=-1)
    doc_norm = np.sqrt(np.sum(doc_embeds ** 2, axis=-1))
    query_norm = np.sqrt(np.sum(query_embed ** 2))
    return dot / (doc_norm * query_norm + 1e-12)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Jina Reranker v3", version="1.0.0")

# Global state (set during startup)
projector: Optional[MLPProjector] = None
start_time: float = 0.0
request_count: int = 0
llama_commit: str = "unknown"


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str] = Field(..., min_length=1)
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True
    return_embeddings: Optional[bool] = False
    instruction: Optional[str] = None


class RerankResponse(BaseModel):
    model: str
    results: List[Dict[str, Any]]
    usage: Dict[str, int]


@app.on_event("startup")
async def startup():
    global projector, start_time, llama_commit

    # Validate required binaries
    for bin_path, label in [
        (LLAMA_EMBEDDING_PATH, "llama-embedding"),
        (LLAMA_TOKENIZE_PATH, "llama-tokenize"),
    ]:
        if not os.path.isfile(bin_path):
            log.error("Required binary not found: %s (%s)", label, bin_path)
            sys.exit(1)
        log.info("Found %s: %s", label, bin_path)

    # Get llama.cpp version
    try:
        proc = await asyncio.create_subprocess_exec(
            LLAMA_EMBEDDING_PATH, "--version",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        version_info = (stdout + stderr).decode("utf-8", errors="replace").strip()
        log.info("llama-embedding version: %s", version_info)
        llama_commit = version_info.split("\n")[0] if version_info else "unknown"
    except Exception as e:
        log.warning("Could not get llama.cpp version: %s", e)

    # Validate model file
    if not os.path.isfile(MODEL_PATH):
        log.error("Model file not found: %s", MODEL_PATH)
        sys.exit(1)
    log.info("Model found: %s", MODEL_PATH)

    # Load projector
    if not os.path.isfile(PROJECTOR_PATH):
        log.error("Projector file not found: %s", PROJECTOR_PATH)
        sys.exit(1)
    projector = load_projector(PROJECTOR_PATH)

    start_time = time.time()
    log.info(
        "Jina Reranker v3 wrapper ready | model=%s | projector=%s | "
        "llama-embedding=%s | ngl=%d | ctx=%d | ubatch=%d",
        MODEL_PATH, PROJECTOR_PATH, LLAMA_EMBEDDING_PATH, NGL, CTX_SIZE, UBATCH_SIZE,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llama_commit": llama_commit,
        "model": MODEL_PATH,
        "projector_loaded": projector is not None,
        "uptime_seconds": time.time() - start_time,
    }


@app.get("/metrics")
async def metrics():
    return {
        "uptime_seconds": time.time() - start_time,
        "request_count": request_count,
    }


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    global request_count
    request_count += 1

    if projector is None:
        raise HTTPException(status_code=503, detail="Projector not loaded (startup incomplete)")

    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    # Format prompt
    prompt = format_rerank_prompt(
        req.query, req.documents, instruction=req.instruction
    )

    # Get per-token hidden states
    try:
        embeddings = await run_llama_embedding(prompt)
    except Exception as e:
        log.error("llama-embedding failed: %s", e)
        raise HTTPException(status_code=502, detail=f"llama-embedding error: {e}")

    # Tokenize to find special token positions
    try:
        tokens = await run_llama_tokenize(prompt)
    except Exception as e:
        log.error("llama-tokenize failed: %s", e)
        raise HTTPException(status_code=502, detail=f"llama-tokenize error: {e}")

    tokens_arr = np.array(tokens, dtype=np.int32)

    query_positions = np.where(tokens_arr == QUERY_EMBED_TOKEN_ID)[0]
    doc_positions = np.where(tokens_arr == DOC_EMBED_TOKEN_ID)[0]

    if len(query_positions) == 0:
        raise HTTPException(
            status_code=500,
            detail=f"Query embed token (ID {QUERY_EMBED_TOKEN_ID}) not found in output",
        )
    if len(doc_positions) == 0:
        raise HTTPException(
            status_code=500,
            detail=f"Document embed tokens (ID {DOC_EMBED_TOKEN_ID}) not found in output",
        )

    if len(doc_positions) != len(req.documents):
        log.warning(
            "Expected %d doc tokens, found %d. Using found positions.",
            len(req.documents), len(doc_positions),
        )

    # Extract embeddings at special token positions
    query_pos = query_positions[0]
    query_hidden = embeddings[query_pos:query_pos + 1]  # [1, hidden_dim]
    doc_hidden = embeddings[doc_positions]  # [n_docs, hidden_dim]

    # Project through MLP
    query_embed = projector(query_hidden)  # [1, 512]
    doc_embeds = projector(doc_hidden)  # [n_docs, 512]

    # Compute scores
    scores = compute_scores(query_embed, doc_embeds)

    # Build results
    results = []
    for idx, (doc, score) in enumerate(zip(req.documents, scores)):
        result: Dict[str, Any] = {
            "index": idx,
            "relevance_score": float(score),
        }
        if req.return_documents:
            result["document"] = doc
        if req.return_embeddings:
            result["embedding"] = doc_embeds[idx].tolist()
        results.append(result)

    # Sort by score descending
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply top_n
    if req.top_n is not None and req.top_n > 0:
        results = results[: req.top_n]

    return RerankResponse(
        model=req.model or "jina/jina-reranker-v3",
        results=results,
        usage={"total_tokens": len(tokens)},
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "jina-reranker-wrapper:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
