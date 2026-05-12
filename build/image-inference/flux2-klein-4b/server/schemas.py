from typing import Literal, Optional

from pydantic import BaseModel, Field, conint


class ImageRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    model: Optional[str] = None
    n: conint(ge=1, le=4) = 1
    size: Literal["768x768", "640x640", "512x512", "384x384"] = "512x512"
    response_format: Literal["b64_json"] = "b64_json"
    seed: Optional[int] = None


class ImageDatum(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int
    data: list[ImageDatum]
