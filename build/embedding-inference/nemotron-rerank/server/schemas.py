from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_DOCUMENTS = 100
MAX_QUERY_LENGTH = 8192
MAX_DOC_LENGTH = 8192


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1)
    top_n: Optional[int] = None

    @field_validator("documents")
    @classmethod
    def _check_documents(cls, v):
        if len(v) > MAX_DOCUMENTS:
            raise ValueError(f"documents must not exceed {MAX_DOCUMENTS}")
        if any(not d.strip() for d in v):
            raise ValueError("empty documents are not allowed")
        return v

    @field_validator("top_n")
    @classmethod
    def _check_top_n(cls, v):
        if v is not None and v < 1:
            raise ValueError("top_n must be >= 1")
        return v


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    object: str = "list"
    model: str
    results: list[RerankResult]
    usage: dict = {}
