"""
OCR processing wrapper around pytesseract.
Accepts PIL Images and returns cleaned text strings.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Thin wrapper around pytesseract for OCR on PIL images.

    Applies basic post-processing to clean up common OCR artifacts.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None) -> None:
        """
        Initialise the OCR processor.

        Args:
            tesseract_cmd: Optional path to the tesseract binary.
                           Falls back to PATH resolution by pytesseract.
        """
        import pytesseract  # noqa: PLC0415

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

        self._pytesseract = pytesseract
        logger.debug("OCRProcessor initialised (tesseract_cmd=%s)", tesseract_cmd)

    def extract_text(self, image: Image.Image) -> str:
        """
        Run OCR on a PIL image and return cleaned text.

        Args:
            image: PIL Image (RGB or RGBA acceptable).

        Returns:
            Cleaned OCR text string. Empty string if no text detected.
        """
        try:
            # Ensure RGB
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            raw = self._pytesseract.image_to_string(
                image,
                config="--psm 6",  # assume uniform block of text
            )
            return self._clean(raw)
        except Exception as exc:
            logger.warning("OCR extraction failed: %s", exc)
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        """Remove excessive whitespace and common OCR artifacts."""
        lines = (line.strip() for line in text.splitlines())
        non_empty = [l for l in lines if l]
        return " ".join(non_empty)
