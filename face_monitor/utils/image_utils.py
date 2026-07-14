"""Image decoding and preprocessing utilities."""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    """Raised when an incoming frame cannot be decoded."""


def decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG payload into a BGR OpenCV image."""

    if "," in data and data.lstrip().startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("Invalid base64 image payload") from exc

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError("Image payload could not be decoded")
    return image


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def to_gray(image_bgr: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def clamp_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

