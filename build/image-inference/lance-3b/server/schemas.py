from typing import Literal, Optional

from pydantic import BaseModel, Field, conint


# Lance generates at 768res for image tasks; we expose the square sizes the
# pipeline maps onto video_height/video_width.
ALLOWED_SIZES = {"768x768", "512x512"}


class ImageRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    model: Optional[str] = None
    n: conint(ge=1, le=4) = 1
    size: Literal["768x768", "512x512"] = "768x768"
    response_format: Literal["b64_json"] = "b64_json"
    seed: Optional[int] = None
    # Non-standard extension: when set, treat request as image edit (Lance
    # image_edit task). base64-encoded PNG/JPEG bytes (no data: prefix). Keeps
    # the JSON body shape so it passes through litellm's image_generation route.
    image_b64: Optional[str] = None


class ImageDatum(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int
    data: list[ImageDatum]
