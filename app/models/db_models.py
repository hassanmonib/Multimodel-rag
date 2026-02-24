"""
SQLAlchemy 2.x ORM models (mapped classes) for the metadata database.
Tables: videos, pdfs, chunks
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


class Video(Base):
    """Metadata record for an ingested YouTube video."""

    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    youtube_url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    channel: Mapped[Optional[str]] = mapped_column(String(256))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(2048))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="video",
        cascade="all, delete-orphan",
        foreign_keys="Chunk.video_id",
    )

    __table_args__ = (Index("ix_videos_youtube_url", "youtube_url"),)


class PDF(Base):
    """Metadata record for an ingested PDF document."""

    __tablename__ = "pdfs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    total_pages: Mapped[Optional[int]] = mapped_column(Integer)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="pdf",
        cascade="all, delete-orphan",
        foreign_keys="Chunk.pdf_id",
    )


class Chunk(Base):
    """
    Unified chunk table for both video and PDF chunks.
    Exactly one of video_id / pdf_id will be set per row.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_type: Mapped[str] = mapped_column(
        String(8), nullable=False
    )  # "video" | "pdf"

    # ── Video-specific ────────────────────────────────────────────────────────
    video_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("videos.id", ondelete="CASCADE"), nullable=True
    )
    time_start: Mapped[Optional[float]] = mapped_column(Float)
    time_end: Mapped[Optional[float]] = mapped_column(Float)
    speaker_label: Mapped[Optional[str]] = mapped_column(String(64))
    speaker_name: Mapped[Optional[str]] = mapped_column(String(256))
    transcript_text: Mapped[Optional[str]] = mapped_column(Text)

    # ── PDF-specific ──────────────────────────────────────────────────────────
    pdf_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("pdfs.id", ondelete="CASCADE"), nullable=True
    )
    page_number: Mapped[Optional[int]] = mapped_column(Integer)
    text_layer: Mapped[Optional[str]] = mapped_column(Text)

    # ── Shared ────────────────────────────────────────────────────────────────
    ocr_text: Mapped[Optional[str]] = mapped_column(Text)
    caption_text: Mapped[Optional[str]] = mapped_column(Text)
    image_path: Mapped[Optional[str]] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    video: Mapped[Optional["Video"]] = relationship(
        "Video", back_populates="chunks", foreign_keys=[video_id]
    )
    pdf: Mapped[Optional["PDF"]] = relationship(
        "PDF", back_populates="chunks", foreign_keys=[pdf_id]
    )

    __table_args__ = (
        Index("ix_chunks_video_id", "video_id"),
        Index("ix_chunks_pdf_id", "pdf_id"),
        Index("ix_chunks_speaker_name", "speaker_name"),
        Index("ix_chunks_chunk_type", "chunk_type"),
    )
