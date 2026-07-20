from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# 공식 API의 encoding_format. OpenAPI 스펙(docs.perplexity.ai/api-reference/
# embeddings-post)의 enum은 base64_int8 / base64_binary 둘뿐이고 기본값은 base64_int8.
#   base64_int8    : signed int8 → base64 (공식, 기본값)
#   base64_binary  : 부호 비트를 packbits → base64 (1024차원 = 128바이트, 공식)
#   base64         : base64_int8의 별칭. OpenAI SDK가 encoding_format 생략 시 자동으로
#                    채워 넣는 값이라, 공식 기본값과 같은 의미를 주도록 맞춘다.
#   base64_float32 : 확장. float32 little-endian → base64 (무손실)
#   float          : 확장. L2 정규화된 float 배열(JSON)
EncodingFormat = Literal[
    "float", "base64", "base64_int8", "base64_binary", "base64_float32"
]

# 공식 문서의 입력 상한.
MAX_INPUTS = 512
# MRL 차원 범위 (0.6b 기준. 4b는 128-2560이지만 우리는 0.6b만 서빙한다).
MIN_DIMENSIONS = 128
MAX_DIMENSIONS = 1024


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str]]
    model: Optional[str] = None
    # 공식 API와 같은 기본값. 나중에 api.perplexity.ai로 폴백할 때 응답 해석이
    # 갈리지 않도록 맞춘다 — 소비 측은 base64 → int8로 디코딩해야 한다.
    # 무손실 float가 필요하면 encoding_format="float"를 명시할 것.
    encoding_format: EncodingFormat = "base64_int8"
    # Matryoshka. None이면 전체 차원(1024).
    dimensions: Optional[int] = Field(None, ge=MIN_DIMENSIONS, le=MAX_DIMENSIONS)

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

    @field_validator("dimensions")
    @classmethod
    def _check_dimensions(cls, v):
        # base64_binary는 packbits라 8의 배수여야 바이트 경계가 맞는다.
        if v is not None and v % 8 != 0:
            raise ValueError("dimensions must be a multiple of 8")
        return v


class EmbeddingDatum(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    # float면 배열, 그 외 인코딩이면 base64 문자열.
    embedding: Union[list[float], str]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: Usage
