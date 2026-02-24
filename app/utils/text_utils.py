"""
Text utility helpers used across the processing pipeline.
"""
from __future__ import annotations

import re
from typing import List


def clean_ocr_text(raw: str) -> str:
    """
    Remove OCR artifacts: excess whitespace, isolated single characters,
    non-printable characters.

    Args:
        raw: Raw string from pytesseract.

    Returns:
        Cleaned string.
    """
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", raw)
    # Collapse multiple whitespace
    text = re.sub(r" {2,}", " ", text)
    # Remove lines that are only punctuation / single chars
    lines = [
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) > 2
    ]
    return " ".join(lines).strip()


def merge_text_fields(*fields: str | None) -> str:
    """
    Concatenate non-empty text fields with a space separator.

    Args:
        *fields: Variable-length text fields (may be None or empty).

    Returns:
        Single merged string.
    """
    return " ".join(f.strip() for f in fields if f and f.strip())


def highlight_query_words(text: str, query: str) -> str:
    """
    Wrap query words in **bold** markdown for Streamlit display.

    Args:
        text: The snippet text to highlight within.
        query: The user's search query.

    Returns:
        Text with matched words wrapped in ** markers.
    """
    words = [re.escape(w) for w in query.split() if len(w) > 2]
    if not words:
        return text
    pattern = re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS or MM:SS string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_snippet(text: str, max_chars: int = 300) -> str:
    """
    Trim text to a readable snippet length without cutting mid-word.

    Args:
        text: Full text string.
        max_chars: Maximum characters to return.

    Returns:
        Trimmed snippet with ellipsis if truncated.
    """
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0]
    return trimmed + " …"
