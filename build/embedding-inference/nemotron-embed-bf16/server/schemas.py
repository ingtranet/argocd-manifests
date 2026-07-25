from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# OpenAI-compatible encoding formats.
# float: L2-normalized float array (JSON, default — matches industry standard)
# base64: base64-encoded float32 bytes (OpenAI SDK default)
EncodingFormat = Literal["float", "base64"]

MAX_INPUTS = 512
# Nemotron-3-Embed-1B hidden dimension = 2048 (from model config.json).
# MRL supported: slice from start and re-normalize.
MODEL_DIMENSIONS = 2048


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = None
    encoding_format: EncodingFormat = "float"
    dimensions: Optional[int] = Field(None, ge=1, le=MODEL_DIMENSIONS)

    @field_validator("input")
    @classmethod
    def _check_input(cls, v):
        texts = [v] if isinstance(v, str) else v
        if not texts:
            raise ValueError("input must not be empty")
        if len(texts) > MAX_INPUTS:
            raise ValueError(f"input must not exceed {MAX_INPUTS} texts")
        if any(not t.strip() for t in texts):
            raise ValueError("empty strings are not allowed")
        return v


class EmbeddingDatum(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[list[float], str]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: Usage
