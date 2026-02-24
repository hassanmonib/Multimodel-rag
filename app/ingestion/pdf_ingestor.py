"""
PDF ingestion pipeline.
Converts pages to images, extracts text layer, runs OCR/captioning,
generates embeddings, and stores everything in PostgreSQL + Qdrant.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Iterator, List, Optional

import fitz  # PyMuPDF

from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import ChunkRepository, PDFRepository
from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk_models import PDFChunk
from app.models.db_models import Chunk, PDF
from app.processing.caption_generator import CaptionGenerator
from app.processing.ocr_processor import OCRProcessor
from app.utils.image_utils import save_image
from app.utils.text_utils import merge_text_fields

logger = logging.getLogger(__name__)


class PDFIngestor:
    """
    End-to-end ingestion pipeline for a PDF file.

    Usage::

        ingestor = PDFIngestor()
        for step, total, message in ingestor.ingest(pdf_bytes, filename):
            print(f"[{step}/{total}] {message}")
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_svc = EmbeddingService()
        self._ocr = OCRProcessor(tesseract_cmd=self._settings.tesseract_cmd)
        self._caption = CaptionGenerator()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(
        self,
        pdf_bytes: bytes,
        filename: str,
    ) -> Iterator[tuple[int, int, str]]:
        """
        Full ingestion pipeline for PDF bytes.  Yields progress updates.

        Args:
            pdf_bytes: Raw bytes of the uploaded PDF.
            filename: Original filename (used as title).

        Yields:
            (current_step, total_steps, message) tuples.
        """
        total_steps = 5
        step = 0

        def _progress(msg: str):
            nonlocal step
            step += 1
            logger.info("[%d/%d] %s", step, total_steps, msg)
            return step, total_steps, msg

        # ── 1. Open PDF ──────────────────────────────────────────────────────
        yield _progress("Opening PDF document …")
        pdf_id = str(uuid.uuid4())
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc.page_count
        title = doc.metadata.get("title") or filename

        # Check duplicate by filename
        with get_session() as session:
            repo = PDFRepository(session)
            if repo.get_by_filename(filename):
                yield step, total_steps, f"'{filename}' already ingested."
                return

        # ── 2. Extract pages ──────────────────────────────────────────────────
        yield _progress(f"Extracting {total_pages} pages …")
        pages = self._extract_pages(doc, pdf_id)
        doc.close()

        # ── 3. OCR + caption ──────────────────────────────────────────────────
        yield _progress("Running OCR and image captioning …")
        enriched = self._enrich_pages(pages)

        # ── 4. Embed + upsert Qdrant ──────────────────────────────────────────
        yield _progress("Generating embeddings and indexing …")
        db_chunks = self._embed_and_store(enriched, pdf_id)

        # ── 5. Persist to PostgreSQL ──────────────────────────────────────────
        yield _progress("Saving to database …")
        pdf_record = PDF(
            id=pdf_id,
            filename=filename,
            title=title,
            total_pages=total_pages,
        )
        with get_session() as session:
            pdf_repo = PDFRepository(session)
            chunk_repo = ChunkRepository(session)
            pdf_repo.create(pdf_record)
            chunk_repo.bulk_create(db_chunks)

        logger.info("PDF ingestion complete: %s – %d chunks", filename, len(db_chunks))
        yield step, total_steps, f"Done! Indexed {total_pages} pages from '{filename}'"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_pages(self, doc: fitz.Document, pdf_id: str) -> List[dict]:
        """
        Render each page to a PIL image and extract the native text layer.

        Returns:
            List of dicts with keys: id, pdf_id, page_number, text_layer, pil_image, image_path.
        """
        from PIL import Image as PILImage  # noqa: PLC0415
        import io  # noqa: PLC0415

        pages = []
        img_dir = str(Path(self._settings.pdf_images_dir) / pdf_id)
        Path(img_dir).mkdir(parents=True, exist_ok=True)

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_num = page_idx + 1
            chunk_id = str(uuid.uuid4())

            # Native text
            text_layer = page.get_text("text").strip() or None

            # Render page to image at 150 DPI
            mat = fitz.Matrix(150 / 72, 150 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

            img_path = str(Path(img_dir) / f"page_{page_num:04d}.jpg")
            save_image(pil_img, img_path)

            pages.append(
                {
                    "id": chunk_id,
                    "pdf_id": pdf_id,
                    "page_number": page_num,
                    "text_layer": text_layer,
                    "pil_image": pil_img,
                    "image_path": img_path,
                }
            )

        return pages

    def _enrich_pages(self, pages: List[dict]) -> List[dict]:
        """Run OCR and caption generation on each page image."""
        for page in pages:
            pil_img = page.get("pil_image")
            if pil_img is None:
                page["ocr_text"] = None
                page["caption_text"] = None
                continue

            try:
                page["ocr_text"] = self._ocr.extract_text(pil_img) or None
            except Exception as exc:
                logger.warning("OCR failed on page %d: %s", page["page_number"], exc)
                page["ocr_text"] = None

            try:
                page["caption_text"] = self._caption.generate(pil_img) or None
            except Exception as exc:
                logger.warning("Caption failed on page %d: %s", page["page_number"], exc)
                page["caption_text"] = None

        return pages

    def _embed_and_store(self, pages: List[dict], pdf_id: str) -> List[Chunk]:
        """
        Generate text + image embeddings for each page and upsert into Qdrant.
        Returns ORM Chunk list for PostgreSQL persistence.
        """
        db_chunks: List[Chunk] = []

        for page in pages:
            merged_text = merge_text_fields(
                page.get("text_layer"),
                page.get("ocr_text"),
                page.get("caption_text"),
            )
            text_vector = self._embedding_svc.embed_text(merged_text or " ")

            image_vector: Optional[List[float]] = None  # type: ignore[type-arg]
            pil_img = page.get("pil_image")
            if pil_img:
                try:
                    image_vector = self._embedding_svc.embed_image(pil_img)
                except Exception as exc:
                    logger.warning("CLIP embedding failed on page %d: %s", page["page_number"], exc)

            metadata = {
                "chunk_type": "pdf",
                "pdf_id": pdf_id,
                "page_number": page["page_number"],
                "text_layer": page.get("text_layer", ""),
                "ocr_text": page.get("ocr_text", ""),
                "caption_text": page.get("caption_text", ""),
                "image_path": page.get("image_path", ""),
            }

            self._embedding_svc.upsert_pdf_chunk(
                chunk_id=page["id"],
                text_vector=text_vector,
                image_vector=image_vector,
                metadata=metadata,
            )

            db_chunks.append(
                Chunk(
                    id=page["id"],
                    chunk_type="pdf",
                    pdf_id=pdf_id,
                    page_number=page["page_number"],
                    text_layer=page.get("text_layer"),
                    ocr_text=page.get("ocr_text"),
                    caption_text=page.get("caption_text"),
                    image_path=page.get("image_path"),
                )
            )

        return db_chunks
