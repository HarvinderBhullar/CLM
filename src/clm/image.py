"""Image states: an image instead of text as the thing a question is asked about.

    from clm import ImageState
    engine.answer(ImageState(path="xray.jpg"), {"fx": Choice(...)}, model="fracatlas")
    client.system_one(ImageState(path="xray.jpg"), ...)    # sent as base64

Wire form: ``{"type": "image", "image": "<base64 or data: URL>"}``. In-process callers may
also pass ``{"type": "image", "path": "..."}``; the HTTP server refuses paths, so a request
cannot make it read files. Only a dict with ``"type": "image"`` is an image state: every
other state (strings, objects, arrays) stays text, exactly as before.

No torch or PIL here, so the client can build image states without them.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from typing import Any


class ImageStateError(ValueError):
    pass


@dataclass
class ImageState:
    path: str | None = None
    image: str | None = None        # base64, or a data: URL

    def __post_init__(self):
        if (self.path is None) == (self.image is None):
            raise ImageStateError("ImageState needs exactly one of path= or image= (base64)")

    def to_dict(self) -> dict:
        """Wire form; a path is read and inlined as base64, so it works over HTTP."""
        if self.path is not None:
            with open(self.path, "rb") as f:
                return {"type": "image", "image": base64.b64encode(f.read()).decode()}
        return {"type": "image", "image": self.image}

    def data(self, allow_path: bool = True) -> bytes:
        """The encoded image file's bytes."""
        if self.path is not None:
            if not allow_path:
                raise ImageStateError("image paths are not accepted here; send the image as base64")
            with open(self.path, "rb") as f:
                return f.read()
        s = self.image.split(",", 1)[1] if self.image.startswith("data:") else self.image
        try:
            return base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ImageStateError(f"image is not valid base64: {e}") from e

    def load(self, allow_path: bool = True):
        """-> (RGB ``PIL.Image``, sha1 of the bytes)."""
        from PIL import Image, UnidentifiedImageError
        raw = self.data(allow_path)
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()
        except (UnidentifiedImageError, OSError) as e:
            raise ImageStateError(f"cannot decode image: {e}") from e
        return im.convert("RGB"), hashlib.sha1(raw).hexdigest()


def as_image_state(state: Any) -> ImageState | None:
    """The state as an ImageState, or None for a text state."""
    if isinstance(state, ImageState):
        return state
    if isinstance(state, dict) and state.get("type") == "image":
        extra = set(state) - {"type", "image", "path"}
        if extra:
            raise ImageStateError(f"image state has unknown fields {sorted(extra)}")
        if not isinstance(state.get("image", ""), str) or not isinstance(state.get("path", ""), str):
            raise ImageStateError("image state fields must be strings")
        return ImageState(path=state.get("path"), image=state.get("image"))
    return None
