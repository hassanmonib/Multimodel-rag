"""
Image captioning using Salesforce BLIP (blip-image-captioning-base).
Heavy model is lazy-loaded and cached as a module-level singleton.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_processor = None
_model = None


def _load_blip(model_name: str) -> None:
    """Load BLIP processor and model into module-level singletons."""
    global _processor, _model
    if _processor is not None:
        return

    from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: PLC0415

    logger.info("Loading BLIP caption model: %s", model_name)
    _processor = BlipProcessor.from_pretrained(model_name)
    _model = BlipForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    _model.eval()
    logger.info("BLIP model loaded on device=%s", device)


class CaptionGenerator:
    """
    Generates natural-language captions for images using BLIP.

    The model is loaded once and reused across all calls.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Args:
            model_name: HuggingFace model ID. Falls back to config value.
        """
        if model_name is None:
            from app.config import get_settings  # noqa: PLC0415

            model_name = get_settings().caption_model
        self._model_name = model_name
        _load_blip(self._model_name)

    def generate(self, image: Image.Image, max_new_tokens: int = 64) -> str:
        """
        Generate a caption for the given PIL image.

        Args:
            image: PIL Image to caption.
            max_new_tokens: Maximum tokens in the output caption.

        Returns:
            Caption string. Empty string on failure.
        """
        global _processor, _model
        try:
            if image.mode not in ("RGB",):
                image = image.convert("RGB")

            device = next(_model.parameters()).device
            inputs = _processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=max_new_tokens)

            caption: str = _processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as exc:
            logger.warning("Caption generation failed: %s", exc)
            return ""
