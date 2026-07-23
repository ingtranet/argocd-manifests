import os
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR", "/models/cstr-llama-nemotron-rerank-1b-v2-ONNX-int8"
)
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4096"))
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))


def cgroup_cpu_limit(root: str = "/sys/fs/cgroup"):
    """Read the CPU limit (cores) imposed on the container. None if unlimited."""
    try:
        with open(f"{root}/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    try:
        with open(f"{root}/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open(f"{root}/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return quota / period if quota > 0 else None
    except (OSError, ValueError):
        return None


def effective_cpu_count(root: str = "/sys/fs/cgroup") -> int:
    host = os.cpu_count() or 1
    limit = cgroup_cpu_limit(root)
    if limit is None:
        return host
    return max(1, min(host, int(limit)))


class NemotronReranker:
    """ONNX Runtime session wrapping the Nemotron reranker INT8 model.

    The model is a cross-encoder: each (query, document) pair is concatenated
    as "[CLS] query [SEP] document [SEP]" and scored. The output logits are
    shape (batch, 2); relevance_score = softmax(logits)[1].
    """

    def __init__(self, session, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self._session = session
        self._tokenizer = tokenizer
        self._max_length = max_length

    @classmethod
    def load(
        cls,
        model_dir: str = DEFAULT_MODEL_DIR,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> "NemotronReranker":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()

        threads = int(os.environ.get("ORT_INTRA_OP_THREADS") or effective_cpu_count())
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.enable_mem_pattern = False

        # CUDA only — no CPU fallback. Fail hard if CUDA is unavailable.
        providers = ["CUDAExecutionProvider"]
        session = ort.InferenceSession(
            f"{model_dir}/model_int8.onnx",
            sess_options=opts,
            providers=providers,
        )
        active_providers = session.get_providers()
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is required but not active. "
                f"Active providers: {active_providers}"
            )

        tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")

        # Enable truncation to max_length for the concatenated pair.
        # The tokenizer handles [CLS] query [SEP] document [SEP] formatting.
        tokenizer.enable_truncation(max_length)

        print(
            f"[nemotron-rerank] loaded {model_dir} "
            f"intra_op_threads={threads} (host={os.cpu_count()}, "
            f"cgroup_limit={cgroup_cpu_limit()}) providers={active_providers} "
            f"max_length={max_length}",
            flush=True,
        )
        return cls(session=session, tokenizer=tokenizer, max_length=max_length)

    def _format_pair(self, query: str, document: str) -> str:
        """Format a query-document pair for the cross-encoder.

        The Nemotron reranker uses the standard cross-encoder format:
        [CLS] query [SEP] document [SEP]
        The tokenizer's template handles this when we pass the pair.
        """
        return query, document

    def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Score all (query, document) pairs and return top_n results.

        Returns list of (index, relevance_score) sorted descending by score.
        """
        results = []

        # Process in batches
        for batch_start in range(0, len(documents), DEFAULT_BATCH_SIZE):
            batch_docs = documents[batch_start : batch_start + DEFAULT_BATCH_SIZE]

            # Encode pairs: the tokenizer handles [CLS] query [SEP] doc [SEP]
            encs = self._tokenizer.encode_batch(
                [(query, doc) for doc in batch_docs]
            )

            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            # Some ONNX models also require token_type_ids
            type_ids = np.array(
                [e.type_ids for e in encs] if hasattr(encs[0], "type_ids") else [e.ids * 0 for e in encs],
                dtype=np.int64,
            )

            feeds = {"input_ids": ids, "attention_mask": mask}
            # Only include token_type_ids if the model expects it
            for inp in self._session.get_inputs():
                if inp.name == "token_type_ids":
                    feeds["token_type_ids"] = type_ids
                    break

            (logits,) = self._session.run(["logits"], feeds)

            # logits shape: (batch, 2) — [non-relevant, relevant]
            # Apply softmax to get probability of relevant class
            exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp / exp.sum(axis=-1, keepdims=True)
            scores = probs[:, 1]  # relevant class probability

            for i, score in enumerate(scores):
                results.append((batch_start + i, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results
