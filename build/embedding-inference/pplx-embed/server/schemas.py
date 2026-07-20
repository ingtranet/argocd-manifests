from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

# 공식 st_quantize.FlexibleQuantizer가 정의한 3종 + 우리가 더한 "float".
# float는 tanh(pooler_output)을 L2 정규화한 무손실 벡터.
Quantization = Literal["float", "int8", "binary", "ubinary"]


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = None
    quantization: Quantization = "float"
    # OpenAI 호환 필드. base64는 미지원이라 float만 받는다.
    encoding_format: Literal["float"] = "float"


class EmbeddingDatum(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: Usage = Field(...)
