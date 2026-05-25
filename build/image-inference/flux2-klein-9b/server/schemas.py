from typing import Literal, Optional

from pydantic import BaseModel, Field, conint


ALLOWED_SIZES = {"1024x1024", "768x768", "640x640", "512x512", "384x384"}


class ImageRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    model: Optional[str] = None
    n: conint(ge=1, le=4) = 1
    size: Literal["1024x1024", "768x768", "640x640", "512x512", "384x384"] = "512x512"
    response_format: Literal["b64_json"] = "b64_json"
    seed: Optional[int] = None
    # Non-standard extension: when set, treat request as image edit.
    # base64-encoded PNG/JPEG bytes (no data: prefix). Must be a string —
    # litellm's image_generation proxy rejects non-string values. For
    # multi-reference editing, join multiple base64 payloads with ',' (the
    # base64 alphabet does not contain ',' so splitting is unambiguous).
    image_b64: Optional[str] = None


class ImageDatum(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int
    data: list[ImageDatum]
