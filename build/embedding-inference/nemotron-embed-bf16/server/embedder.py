"""
BF16 embedding server for nvidia/Nemotron-3-Embed-1B-BF16.

Loads the model via HuggingFace transformers with torch.bfloat16.
Requires CUDA — no CPU fallback.

Pooling strategy: average pooling (confirmed from model config.json
`"pooling": "avg"` and model card: "average pooling to the transformer's
token-level representations"). L2 normalization applied after pooling.

The model supports dynamic embedding sizes (Matryoshka / MRL) by slicing
from the start and re-normalizing, as documented in the model card.

Prefix convention (from model card):
  - queries: "query: <text>"
  - passages: "passage: <text>"
The server does NOT auto-prefix — the caller must add the appropriate
prefix. This matches the sentence-transformers behavior where the
model's `1_Pooling` config stores the prompts.
"""
import base64
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL_ID = os.environ.get(
    "MODEL_ID", "nvidia/Nemotron-3-Embed-1B-BF16"
)
DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "")
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "32768"))
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))


def postprocess(embeddings: np.ndarray, encoding_format: str):
    """Convert numpy embeddings to the requested encoding format.

    embeddings is already L2-normalized float32.
    """
    if encoding_format == "float":
        return [row.tolist() for row in embeddings]

    if encoding_format == "base64":
        return [base64.b64encode(row.tobytes()).decode() for row in embeddings]

    raise ValueError(f"Invalid encoding_format: {encoding_format}")


class NemotronEmbedder:
    """HuggingFace transformers model wrapper for BF16 embedding."""

    def __init__(self, model, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = model.device

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        model_dir: str = DEFAULT_MODEL_DIR,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> "NemotronEmbedder":
        from transformers import AutoModel, AutoTokenizer

        # If MODEL_DIR is set, load from local path (initContainer download).
        # Otherwise, load from HuggingFace hub.
        load_path = model_dir if model_dir else model_id

        print(
            f"[nemotron-embed] loading {load_path} "
            f"with torch.bfloat16 ...",
            flush=True,
        )

        model = AutoModel.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Explicitly move to CUDA. Single-GPU server: no device_map needed
        # (device_map triggers accelerate dependency).
        model = model.to(torch.device("cuda"))

        # Verify CUDA is active
        if not next(model.parameters()).is_cuda:
            raise RuntimeError(
                "Model is not on CUDA device. BF16 inference requires CUDA. "
                "Check CUDA availability."
            )

        print(
            f"[nemotron-embed] loaded {load_path} "
            f"device={model.device} "
            f"dtype={next(model.parameters()).dtype} "
            f"max_length={max_length}",
            flush=True,
        )
        return cls(model=model, tokenizer=tokenizer, max_length=max_length)

    def _pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Average pooling over non-padding tokens + L2 normalization.

        Confirmed from model config.json: `"pooling": "avg"` and model card:
        "the final embedding vector is obtained by applying average pooling
        to the transformer's token-level representations."
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        lengths = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return summed / lengths

    def embed(
        self,
        texts: list[str],
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ):
        """(encoded_embeddings_list, total_non_padding_tokens)"""
        # Tokenize
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        # Forward pass
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

        # Average pooling (confirmed from model config.json: pooling=avg)
        embeddings = self._pool(outputs.last_hidden_state, attention_mask)

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Optional dimension truncation (Matryoshka / MRL)
        if dimensions is not None:
            if dimensions < embeddings.size(-1):
                embeddings = embeddings[:, :dimensions]
                # Re-normalize after truncation (per model card recommendation)
                embeddings = F.normalize(embeddings, p=2, dim=1)

        # Count non-padding tokens
        total_tokens = int(attention_mask.sum().item())

        # Convert to float32 numpy for postprocessing
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        return postprocess(embeddings_np, encoding_format), total_tokens
