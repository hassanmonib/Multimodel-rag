"""
Application configuration via Pydantic Settings.
All values sourced from environment variables / .env file.
"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object. Values are read from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./multimodal_rag.db",
        description="SQLAlchemy database URL",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant Cloud URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant Cloud API key")
    qdrant_video_collection: str = Field(default="video_chunks")
    qdrant_pdf_collection: str = Field(default="pdf_chunks")

    # ── OpenAI / LLM ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_base_url: Optional[str] = Field(default=None)

    # ── HuggingFace ───────────────────────────────────────────────────────────
    huggingface_token: Optional[str] = Field(default=None)

    # ── Models ────────────────────────────────────────────────────────────────
    text_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    clip_model: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="openai")
    caption_model: str = Field(default="Salesforce/blip-image-captioning-base")
    whisper_model_size: str = Field(default="base")

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_dir: str = Field(default="./storage")
    video_frames_dir: str = Field(default="./storage/frames")
    pdf_images_dir: str = Field(default="./storage/pdf_images")

    # ── Tesseract ─────────────────────────────────────────────────────────────
    tesseract_cmd: Optional[str] = Field(default=None)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5)
    default_confidence_threshold: float = Field(default=0.3)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    def ensure_storage_dirs(self) -> None:
        """Create storage directories if they don't exist."""
        for path in [self.storage_dir, self.video_frames_dir, self.pdf_images_dir]:
            os.makedirs(path, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_storage_dirs()
    return _settings
