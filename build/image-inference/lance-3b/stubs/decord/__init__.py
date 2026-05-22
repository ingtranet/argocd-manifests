"""Import-only stub for `decord`.

`decord` has no aarch64/py3.12 wheel, but Lance imports it at the top of
data/datasets_custom/validation_dataset.py. Only the video tasks ever call into
it; this image serves image tasks (t2i, image_edit) only, so we satisfy the
import with a stub that raises loudly if any video-decode path is hit.
"""

_MSG = (
    "decord is a stub in this image (no aarch64 wheel). Video tasks are not "
    "supported by the lance-3b image-only build."
)


class VideoReader:  # noqa: N801 - mirror decord's public name
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)


class _Ctx:
    def __init__(self, *args, **kwargs):
        pass


def cpu(*args, **kwargs):
    return _Ctx()


def gpu(*args, **kwargs):
    return _Ctx()


class _Bridge:
    @staticmethod
    def set_bridge(*args, **kwargs):
        pass


bridge = _Bridge()
