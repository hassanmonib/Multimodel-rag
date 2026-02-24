"""
Repository classes for CRUD operations on Video, PDF, and Chunk ORM models.
All mutations are session-scoped; callers use get_session() context manager.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import Chunk, PDF, Video

logger = logging.getLogger(__name__)


# ── Video Repository ──────────────────────────────────────────────────────────

class VideoRepository:
    """CRUD for the Video table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, video: Video) -> Video:
        """Persist a new Video record."""
        self._session.add(video)
        self._session.flush()
        logger.debug("Created Video id=%s", video.id)
        return video

    def get_by_id(self, video_id: str) -> Optional[Video]:
        """Return a Video by primary key, or None."""
        return self._session.get(Video, video_id)

    def get_by_url(self, youtube_url: str) -> Optional[Video]:
        """Return a Video matching the given YouTube URL, or None."""
        stmt = select(Video).where(Video.youtube_url == youtube_url)
        return self._session.scalar(stmt)

    def list_all(self) -> List[Video]:
        """Return all ingested videos."""
        return list(self._session.scalars(select(Video)).all())

    def delete(self, video_id: str) -> bool:
        """Delete a Video and cascade to its chunks. Returns True if found."""
        video = self.get_by_id(video_id)
        if video is None:
            return False
        self._session.delete(video)
        logger.debug("Deleted Video id=%s", video_id)
        return True


# ── PDF Repository ────────────────────────────────────────────────────────────

class PDFRepository:
    """CRUD for the PDF table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, pdf: PDF) -> PDF:
        """Persist a new PDF record."""
        self._session.add(pdf)
        self._session.flush()
        logger.debug("Created PDF id=%s", pdf.id)
        return pdf

    def get_by_id(self, pdf_id: str) -> Optional[PDF]:
        """Return a PDF by primary key, or None."""
        return self._session.get(PDF, pdf_id)

    def get_by_filename(self, filename: str) -> Optional[PDF]:
        """Return the most-recently ingested PDF with the given filename."""
        stmt = (
            select(PDF)
            .where(PDF.filename == filename)
            .order_by(PDF.ingested_at.desc())
            .limit(1)
        )
        return self._session.scalar(stmt)

    def list_all(self) -> List[PDF]:
        """Return all ingested PDFs."""
        return list(self._session.scalars(select(PDF)).all())

    def delete(self, pdf_id: str) -> bool:
        """Delete a PDF and cascade to its chunks. Returns True if found."""
        pdf = self.get_by_id(pdf_id)
        if pdf is None:
            return False
        self._session.delete(pdf)
        logger.debug("Deleted PDF id=%s", pdf_id)
        return True


# ── Chunk Repository ──────────────────────────────────────────────────────────

class ChunkRepository:
    """CRUD for the unified Chunk table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, chunk: Chunk) -> Chunk:
        """Persist a new Chunk record."""
        self._session.add(chunk)
        self._session.flush()
        return chunk

    def bulk_create(self, chunks: List[Chunk]) -> None:
        """Efficiently persist many chunks at once."""
        self._session.add_all(chunks)
        self._session.flush()
        logger.debug("Bulk created %d chunks", len(chunks))

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Return a Chunk by primary key, or None."""
        return self._session.get(Chunk, chunk_id)

    def get_by_video(self, video_id: str) -> List[Chunk]:
        """Return all chunks for a given video, ordered by time_start."""
        stmt = (
            select(Chunk)
            .where(Chunk.video_id == video_id)
            .order_by(Chunk.time_start)
        )
        return list(self._session.scalars(stmt).all())

    def get_by_pdf(self, pdf_id: str) -> List[Chunk]:
        """Return all chunks for a given PDF, ordered by page_number."""
        stmt = (
            select(Chunk)
            .where(Chunk.pdf_id == pdf_id)
            .order_by(Chunk.page_number)
        )
        return list(self._session.scalars(stmt).all())

    def list_speakers(self) -> List[str]:
        """Return distinct non-null speaker names across all video chunks."""
        from sqlalchemy import distinct

        stmt = select(distinct(Chunk.speaker_name)).where(
            Chunk.speaker_name.isnot(None)
        )
        return [row for row in self._session.scalars(stmt).all() if row]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """Fetch multiple chunks by their IDs in one query."""
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        return list(self._session.scalars(stmt).all())
