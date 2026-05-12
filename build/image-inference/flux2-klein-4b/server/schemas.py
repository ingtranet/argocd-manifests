from typing import Literal, Optional

from pydantic import BaseModel, Field, conint


class ImageRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    model: Optional[str] = None
    n: conint(ge=1, le=4) = 1
    size: Literal["1024x1024", "768x768", "512x512"] = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"
    seed: Optional[int] = None


class ImageDatum(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int
    data: list[ImageDatum]
