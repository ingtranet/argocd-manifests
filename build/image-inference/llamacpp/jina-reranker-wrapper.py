#!/usr/bin/env python3
"""
Jina Reranker v3 HTTP wrapper for llama.cpp on Jetson Orin.

Implements POST /v1/rerank (standard OpenAI-compatible rerank schema),
GET  /health, and GET /metrics.

Key robustness features:
  - Batches documents into groups whose prompt fits within CTX_SIZE (8192).
  - Enforces a maximum of 64 documents per request.
  - Serializes llama subprocess calls via an asyncio semaphore to prevent
    Jetson memory contention.
  - Applies a subprocess timeout so hung llama processes do not stall the
    server indefinitely.
  - Returns clear 4xx/5xx errors instead of crashing the server.

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
from typing import List, Optional, Dict, Any, Tuple

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
# Maximum documents per request (Jina v3 capability ceiling)
MAX_DOCUMENTS = int(os.environ.get("MAX_DOCUMENTS", "64"))
# Token budget headroom: leave this many tokens free below CTX_SIZE for
# the prompt prefix, suffix, query, and special tokens.
TOKEN_HEADROOM = int(os.environ.get("TOKEN_HEADROOM", "256"))
# Subprocess timeout in seconds for each llama-embedding call
LLAMA_TIMEOUT = int(os.environ.get("LLAMA_TIMEOUT", "300"))

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
# Token estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for most languages."""
    return max(1, len(text) // 4)


def estimate_prompt_tokens(
    query: str,
    n_docs: int,
    avg_doc_chars: int,
    instruction: Optional[str] = None,
) -> int:
    """Estimate total token count for a rerank prompt with n_docs documents."""
    # Fixed overhead: prefix + suffix + query wrapper + doc wrapper per doc
    # Prefix ~120 tokens, suffix ~5 tokens, query wrapper ~10 tokens
    # Each doc wrapper: '<passage id="N">\n' + '\n</passage>' + DOC_EMBED_TOKEN
    #   ~15 tokens overhead per doc
    fixed = 135  # prefix + suffix
    query_tokens = estimate_tokens(query)
    instr_tokens = estimate_tokens(instruction) if instruction else 0
    per_doc_overhead = 15  # passage wrapper tokens
    per_doc_content = avg_doc_chars // 4
    return fixed + query_tokens + instr_tokens + n_docs * (per_doc_overhead + per_doc_content)


# ---------------------------------------------------------------------------
# llama-embedding CLI wrapper (with semaphore + timeout)
# ---------------------------------------------------------------------------
# Semaphore to serialize llama subprocess calls (Jetson memory contention)
_llama_semaphore = asyncio.Semaphore(1)


async def run_llama_embedding(prompt: str) -> np.ndarray:
    """Run llama-embedding and return per-token hidden states as [n_tokens, hidden_dim]."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", prefix="jina_prompt_"
    ) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        async with _llama_semaphore:
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
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=LLAMA_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(
                    f"llama-embedding timed out after {LLAMA_TIMEOUT}s"
                )

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
        async with _llama_semaphore:
            proc = await asyncio.create_subprocess_exec(
                LLAMA_TOKENIZE_PATH,
                "-m", MODEL_PATH,
                "-f", prompt_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError("llama-tokenize timed out")

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


def _extract_embeddings(
    embeddings: np.ndarray,
    tokens: List[int],
    n_expected_docs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract query and document hidden states at special token positions.

    Returns (query_hidden [1, hidden_dim], doc_hidden [n_docs, hidden_dim]).
    """
    tokens_arr = np.array(tokens, dtype=np.int32)
    query_positions = np.where(tokens_arr == QUERY_EMBED_TOKEN_ID)[0]
    doc_positions = np.where(tokens_arr == DOC_EMBED_TOKEN_ID)[0]

    if len(query_positions) == 0:
        raise ValueError(
            f"Query embed token (ID {QUERY_EMBED_TOKEN_ID}) not found in output"
        )
    if len(doc_positions) == 0:
        raise ValueError(
            f"Document embed tokens (ID {DOC_EMBED_TOKEN_ID}) not found in output"
        )

    if len(doc_positions) != n_expected_docs:
        log.warning(
            "Expected %d doc tokens, found %d. Using found positions.",
            n_expected_docs, len(doc_positions),
        )

    query_pos = query_positions[0]
    query_hidden = embeddings[query_pos:query_pos + 1]  # [1, hidden_dim]
    doc_hidden = embeddings[doc_positions]  # [n_docs, hidden_dim]
    return query_hidden, doc_hidden


async def _score_batch(
    query: str,
    documents: List[str],
    global_indices: List[int],
    instruction: Optional[str] = None,
    return_documents: bool = True,
) -> List[Dict[str, Any]]:
    """Score one batch of documents and return partial results.

    Each result dict has keys: index, relevance_score, document (if requested).
    """
    prompt = format_rerank_prompt(query, documents, instruction=instruction)

    # Get per-token hidden states
    embeddings = await run_llama_embedding(prompt)

    # Tokenize to find special token positions
    tokens = await run_llama_tokenize(prompt)

    query_hidden, doc_hidden = _extract_embeddings(embeddings, tokens, len(documents))

    # Project through MLP
    query_embed = projector(query_hidden)  # [1, 512]
    doc_embeds = projector(doc_hidden)  # [n_docs, 512]

    # Compute scores
    scores = compute_scores(query_embed, doc_embeds)

    results = []
    for local_idx, (doc, score) in enumerate(zip(documents, scores)):
        result: Dict[str, Any] = {
            "index": global_indices[local_idx],
            "relevance_score": float(score),
        }
        if return_documents:
            result["document"] = {"text": doc}
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Jina Reranker v3", version="1.1.0")

# Global state (set during startup)
projector: Optional[MLPProjector] = None
start_time: float = 0.0
request_count: int = 0
llama_commit: str = "unknown"
# Track whether the request asked for return_documents (set per-request)


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
        "llama-embedding=%s | ngl=%d | ctx=%d | ubatch=%d | max_docs=%d | "
        "headroom=%d | timeout=%ds",
        MODEL_PATH, PROJECTOR_PATH, LLAMA_EMBEDDING_PATH, NGL, CTX_SIZE,
        UBATCH_SIZE, MAX_DOCUMENTS, TOKEN_HEADROOM, LLAMA_TIMEOUT,
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

    # Enforce maximum document count
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents: {len(req.documents)} exceeds maximum {MAX_DOCUMENTS}",
        )

    # Check for oversized individual documents (would exceed context even alone)
    max_doc_tokens = CTX_SIZE - TOKEN_HEADROOM - 200  # 200 for query + overhead
    for i, doc in enumerate(req.documents):
        doc_tokens = estimate_tokens(doc)
        if doc_tokens > max_doc_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Document {i} is too large: estimated {doc_tokens} tokens "
                       f"(max {max_doc_tokens})",
            )

    return_documents = bool(req.return_documents)

    # Compute average document character length for token estimation
    avg_doc_chars = sum(len(d) for d in req.documents) // max(1, len(req.documents))

    # Determine batch size: how many documents fit in one prompt?
    # Binary search for the largest batch that stays within budget
    budget = CTX_SIZE - TOKEN_HEADROOM
    n_docs = len(req.documents)

    def batch_fits(k: int) -> bool:
        return estimate_prompt_tokens(req.query, k, avg_doc_chars, req.instruction) <= budget

    # Find max docs per batch
    if n_docs <= 1 or batch_fits(n_docs):
        # Single batch
        batches = [(req.documents, list(range(n_docs)))]
    else:
        # Find batch size via binary search
        lo, hi = 1, n_docs
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if batch_fits(mid):
                lo = mid
            else:
                hi = mid - 1
        batch_size = lo
        log.info(
            "Batching %d documents into groups of ~%d (budget=%d tokens)",
            n_docs, batch_size, budget,
        )
        batches = []
        for start in range(0, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            batches.append((req.documents[start:end], list(range(start, end))))

    # Score each batch
    all_results: List[Dict[str, Any]] = []
    total_tokens = 0
    for batch_docs, global_indices in batches:
        try:
            batch_results = await _score_batch(
                req.query, batch_docs, global_indices,
                instruction=req.instruction,
                return_documents=return_documents,
            )
            all_results.extend(batch_results)
            # Estimate tokens from the first batch's token count
            # (we don't have exact per-batch token counts without re-tokenizing)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except RuntimeError as e:
            log.error("llama subprocess error in batch: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            log.error("Unexpected error in batch: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Sort by score descending
    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Apply top_n
    if req.top_n is not None and req.top_n > 0:
        all_results = all_results[: req.top_n]

    return RerankResponse(
        model=req.model or "jina/jina-reranker-v3",
        results=all_results,
        usage={"total_tokens": total_tokens or 0},
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
