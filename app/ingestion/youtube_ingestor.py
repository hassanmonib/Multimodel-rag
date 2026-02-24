"""
YouTube ingestion pipeline.
Downloads, transcribes, diarizes, chunks, extracts frames, runs OCR/captioning,
generates embeddings, and stores everything in PostgreSQL + Qdrant.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from app.config import get_settings
from app.database.connection import get_session
from app.database.repositories import ChunkRepository, VideoRepository
from app.embeddings.embedding_service import EmbeddingService
from app.models.chunk_models import VideoChunk
from app.models.db_models import Chunk, Video
from app.processing.caption_generator import CaptionGenerator
from app.processing.diarization_processor import DiarizationProcessor
from app.processing.keyframe_extractor import KeyframeExtractor
from app.processing.ocr_processor import OCRProcessor
from app.services.deepgram_service import transcribe_with_diarization
from app.services.youtube_service import download_video, get_video_metadata
from app.utils.image_utils import save_image
from app.utils.text_utils import merge_text_fields

logger = logging.getLogger(__name__)

# Configurable chunk window size in seconds
CHUNK_WINDOW_SECS = 15.0


class YouTubeIngestor:
    """
    End-to-end ingestion pipeline for a YouTube video URL.

    Usage::

        ingestor = YouTubeIngestor()
        for step, total, message in ingestor.ingest(url):
            print(f"[{step}/{total}] {message}")
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_svc = EmbeddingService()
        self._ocr = OCRProcessor(tesseract_cmd=self._settings.tesseract_cmd)
        self._caption = CaptionGenerator()
        self._keyframe = KeyframeExtractor()
        self._diarizer = DiarizationProcessor(
            hf_token=self._settings.huggingface_token
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, youtube_url: str) -> Iterator[tuple[int, int, str]]:
        """
        Full ingestion pipeline as a generator yielding progress updates.

        Yields:
            (current_step, total_steps, message) tuples.

        Args:
            youtube_url: Public YouTube video URL.
        """
        use_deepgram = bool(self._settings.deepgram_api_key)
        total_steps = 7 if use_deepgram else 8
        step = 0

        def _progress(msg: str):
            nonlocal step
            step += 1
            logger.info("[%d/%d] %s", step, total_steps, msg)
            return step, total_steps, msg

        # ── 1. Fetch metadata ─────────────────────────────────────────────────
        yield _progress("Fetching video metadata …")
        meta = get_video_metadata(youtube_url)
        video_id = meta["id"]

        # Check if already ingested
        with get_session() as session:
            repo = VideoRepository(session)
            existing = repo.get_by_url(youtube_url)
            if existing:
                yield step, total_steps, f"Already ingested: {meta['title']}"
                return

        # ── 2. Download video + audio ─────────────────────────────────────────
        yield _progress("Downloading video and audio …")
        dl_dir = str(Path(self._settings.storage_dir) / "downloads" / video_id)
        paths = download_video(youtube_url, dl_dir, video_id=video_id)
        video_path = paths["video_path"]
        audio_path = paths["audio_path"]

        # ── 3. Transcribe (and diarize if using Deepgram) ──────────────────────
        if use_deepgram:
            yield _progress("Transcribing with Deepgram (with diarization) …")
            transcript_segments, diarization_segs = transcribe_with_diarization(
                audio_path,
                self._settings.deepgram_api_key,
                model=self._settings.deepgram_model,
            )
        else:
            yield _progress("Transcribing with Whisper …")
            transcript_segments = self._transcribe(audio_path)
            yield _progress("Running speaker diarization …")
            diarization_segs = self._diarizer.diarize(audio_path)

        # ── 4. Build chunks ───────────────────────────────────────────────────
        yield _progress("Building transcript chunks …")
        raw_chunks = self._build_chunks(transcript_segments, diarization_segs, video_id)

        # ── 5. Extract keyframes ──────────────────────────────────────────────
        yield _progress("Extracting keyframes …")
        keyframes = self._keyframe.extract(video_path)
        frame_dir = str(Path(self._settings.video_frames_dir) / video_id)
        os.makedirs(frame_dir, exist_ok=True)

        # Assign nearest keyframe to each chunk
        raw_chunks = self._assign_keyframes(raw_chunks, keyframes, frame_dir, video_id)

        # ── 6. OCR + caption + embed ──────────────────────────────────────────
        yield _progress("Running OCR, captioning, and embedding …")
        db_chunks, pydantic_chunks = self._enrich_and_embed(raw_chunks, video_id, youtube_url)

        # ── 7. Persist to PostgreSQL ──────────────────────────────────────────
        yield _progress("Saving to database …")
        video_record = Video(
            id=video_id,
            youtube_url=youtube_url,
            title=meta.get("title", ""),
            channel=meta.get("channel", ""),
            duration_seconds=meta.get("duration"),
            thumbnail_url=meta.get("thumbnail", ""),
        )
        with get_session() as session:
            video_repo = VideoRepository(session)
            chunk_repo = ChunkRepository(session)
            video_repo.create(video_record)
            chunk_repo.bulk_create(db_chunks)

        logger.info("Ingestion complete for video %s – %d chunks", video_id, len(db_chunks))
        yield step, total_steps, f"Done! Ingested {len(db_chunks)} chunks from '{meta['title']}'"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _transcribe(self, audio_path: str) -> List[Dict]:
        """Run Whisper transcription and return word-level segment dicts."""
        import whisper  # noqa: PLC0415

        logger.info("Whisper model: %s", self._settings.whisper_model_size)
        model = whisper.load_model(self._settings.whisper_model_size)
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False,
        )
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                {
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
        logger.info("Transcribed %d segments", len(segments))
        return segments

    def _build_chunks(
        self,
        segments: List[Dict],
        diarization_segs: List,
        video_id: str,
    ) -> List[Dict]:
        """Group transcript segments into fixed-window chunks."""
        chunks: List[Dict] = []
        if not segments:
            return chunks

        window_start = segments[0]["start"]
        window_texts: List[str] = []
        window_end = window_start

        for seg in segments:
            window_texts.append(seg["text"])
            window_end = seg["end"]

            if window_end - window_start >= CHUNK_WINDOW_SECS:
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    {
                        "id": chunk_id,
                        "video_id": video_id,
                        "time_start": window_start,
                        "time_end": window_end,
                        "transcript_text": " ".join(window_texts),
                        "speaker_label": None,
                    }
                )
                window_start = window_end
                window_texts = []

        # Last partial window
        if window_texts:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "video_id": video_id,
                    "time_start": window_start,
                    "time_end": window_end,
                    "transcript_text": " ".join(window_texts),
                    "speaker_label": None,
                }
            )

        if diarization_segs:
            chunks = self._diarizer.assign_speakers(chunks, diarization_segs)

        logger.info("Built %d transcript chunks", len(chunks))
        return chunks

    def _assign_keyframes(
        self,
        chunks: List[Dict],
        keyframes: List,
        frame_dir: str,
        video_id: str,
    ) -> List[Dict]:
        """
        For each chunk, find the nearest keyframe by timestamp and
        save it to disk. Attach image_path to the chunk dict.
        """
        for chunk in chunks:
            mid_time = (chunk["time_start"] + chunk["time_end"]) / 2.0
            best_kf = min(keyframes, key=lambda kf: abs(kf[0] - mid_time)) if keyframes else None
            if best_kf:
                frame_ts, frame_img = best_kf
                img_path = str(Path(frame_dir) / f"{chunk['id']}.jpg")
                save_image(frame_img, img_path)
                chunk["image_path"] = img_path
            else:
                chunk["image_path"] = None
        return chunks

    def _enrich_and_embed(
        self,
        chunks: List[Dict],
        video_id: str,
        youtube_url: str,
    ) -> tuple[List[Chunk], List[VideoChunk]]:
        """
        For each raw chunk dict:
        - Run OCR on keyframe image
        - Generate caption
        - Merge text fields and embed
        - Embed keyframe image with CLIP
        - Upsert into Qdrant
        Returns (ORM Chunk list, Pydantic VideoChunk list).
        """
        from PIL import Image as PILImage  # noqa: PLC0415

        db_chunks: List[Chunk] = []
        pydantic_chunks: List[VideoChunk] = []

        for chunk in chunks:
            img_path = chunk.get("image_path")
            pil_img = None
            ocr_text = ""
            caption_text = ""
            image_vector = None

            if img_path and Path(img_path).is_file():
                try:
                    pil_img = PILImage.open(img_path).convert("RGB")
                    ocr_text = self._ocr.extract_text(pil_img)
                    caption_text = self._caption.generate(pil_img)
                    image_vector = self._embedding_svc.embed_image(pil_img)
                except Exception as exc:
                    logger.warning("Media processing failed for chunk %s: %s", chunk["id"], exc)

            merged_text = merge_text_fields(
                chunk.get("transcript_text"), ocr_text, caption_text
            )
            text_vector = self._embedding_svc.embed_text(merged_text)

            metadata = {
                "chunk_type": "video",
                "video_id": video_id,
                "youtube_url": youtube_url,
                "time_start": chunk["time_start"],
                "time_end": chunk["time_end"],
                "speaker_label": chunk.get("speaker_label"),
                "speaker_name": chunk.get("speaker_label"),  # label == name until resolved
                "transcript_text": chunk.get("transcript_text", ""),
                "ocr_text": ocr_text,
                "caption_text": caption_text,
                "image_path": img_path or "",
            }

            self._embedding_svc.upsert_video_chunk(
                chunk_id=chunk["id"],
                text_vector=text_vector,
                image_vector=image_vector,
                metadata=metadata,
            )

            db_chunk = Chunk(
                id=chunk["id"],
                chunk_type="video",
                video_id=video_id,
                time_start=chunk["time_start"],
                time_end=chunk["time_end"],
                speaker_label=chunk.get("speaker_label"),
                speaker_name=chunk.get("speaker_label"),
                transcript_text=chunk.get("transcript_text", ""),
                ocr_text=ocr_text,
                caption_text=caption_text,
                image_path=img_path,
            )
            db_chunks.append(db_chunk)

            pydantic_chunks.append(
                VideoChunk(
                    id=chunk["id"],
                    video_id=video_id,
                    time_start=chunk["time_start"],
                    time_end=chunk["time_end"],
                    transcript_text=chunk.get("transcript_text", ""),
                    speaker_label=chunk.get("speaker_label"),
                    speaker_name=chunk.get("speaker_label"),
                    ocr_text=ocr_text or None,
                    caption_text=caption_text or None,
                    image_path=img_path,
                )
            )

        return db_chunks, pydantic_chunks
