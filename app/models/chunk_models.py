"""
Pydantic v2 data models for video and PDF chunks.
These are used throughout ingestion, retrieval, and display layers.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VideoChunk(BaseModel):
    """
    Represents a single time-windowed chunk of a YouTube video.

    Attributes:
        id: Unique chunk identifier (UUID4 string).
        video_id: Foreign key referencing the parent Video record.
        time_start: Start timestamp in seconds.
        time_end: End timestamp in seconds.
        transcript_text: Whisper-transcribed speech for this window.
        speaker_label: Raw diarization label (e.g. "SPEAKER_00").
        speaker_name: Resolved human-readable speaker name if available.
        ocr_text: Text extracted via OCR from the keyframe image.
        caption_text: BLIP-generated caption of the keyframe image.
        image_path: Absolute path to the saved keyframe image on disk.
    """

    id: str = Field(..., description="UUID4 chunk identifier")
    video_id: str = Field(..., description="Parent video identifier")
    time_start: float = Field(..., ge=0.0, description="Chunk start time (seconds)")
    time_end: float = Field(..., ge=0.0, description="Chunk end time (seconds)")
    transcript_text: str = Field(..., description="Transcribed speech text")
    speaker_label: Optional[str] = Field(default=None, description="Raw diarization label")
    speaker_name: Optional[str] = Field(default=None, description="Human-readable speaker name")
    ocr_text: Optional[str] = Field(default=None, description="OCR text from keyframe")
    caption_text: Optional[str] = Field(default=None, description="BLIP caption of keyframe")
    image_path: Optional[str] = Field(default=None, description="Path to keyframe image on disk")


class PDFChunk(BaseModel):
    """
    Represents a single page/slide chunk of a PDF document.

    Attributes:
        id: Unique chunk identifier (UUID4 string).
        pdf_id: Foreign key referencing the parent PDF record.
        page_number: 1-indexed page number within the PDF.
        text_layer: Native text extracted by PyMuPDF (may be empty for scanned PDFs).
        ocr_text: Text extracted via OCR from the page image.
        caption_text: BLIP-generated caption for the slide image.
        image_path: Absolute path to the saved page image on disk.
    """

    id: str = Field(..., description="UUID4 chunk identifier")
    pdf_id: str = Field(..., description="Parent PDF identifier")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text_layer: Optional[str] = Field(default=None, description="Native PDF text layer")
    ocr_text: Optional[str] = Field(default=None, description="OCR text from page image")
    caption_text: Optional[str] = Field(default=None, description="BLIP caption of slide")
    image_path: Optional[str] = Field(default=None, description="Path to page image on disk")
