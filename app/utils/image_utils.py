"""
Image utility helpers: saving, loading, encoding, and resizing.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def save_image(image: Image.Image, path: str) -> str:
    """
    Save a PIL Image to disk as JPEG, creating parent directories as needed.

    Args:
        image: PIL Image to save.
        path: Absolute file path (must end with .jpg or .jpeg).

    Returns:
        The resolved absolute path string.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=90, optimize=True)
    logger.debug("Saved image: %s", path)
    return path


def load_image(path: str) -> Optional[Image.Image]:
    """
    Load an image from disk.

    Args:
        path: Absolute path to image file.

    Returns:
        PIL Image, or None if the file does not exist / cannot be read.
    """
    if not Path(path).is_file():
        logger.warning("Image not found: %s", path)
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.error("Failed to load image %s: %s", path, exc)
        return None


def load_image_as_base64(path: str) -> Optional[str]:
    """
    Load an image from disk and return it as a base64-encoded JPEG string.
    Useful for embedding in HTML/Streamlit.

    Args:
        path: Absolute path to image file.

    Returns:
        Base64 string, or None if loading fails.
    """
    img = load_image(path)
    if img is None:
        return None
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_for_display(
    image: Image.Image,
    max_width: int = 800,
    max_height: int = 600,
) -> Image.Image:
    """
    Resize an image while preserving aspect ratio to fit within max dimensions.

    Args:
        image: PIL Image.
        max_width: Maximum output width in pixels.
        max_height: Maximum output height in pixels.

    Returns:
        Resized PIL Image (may be unchanged if already fits).
    """
    w, h = image.size
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL Image to raw bytes in the given format."""
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format=fmt)
    return buf.getvalue()
