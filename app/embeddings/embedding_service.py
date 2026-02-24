"""
Embedding service: generates text embeddings (sentence-transformers) and
image embeddings (CLIP), then upserts them into Qdrant with full metadata.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level singletons
_text_model = None
_clip_model = None
_clip_preprocess = None


def _load_text_model(model_name: str):
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Loading text embedding model: %s", model_name)
        _text_model = SentenceTransformer(model_name)
    return _text_model


def _load_clip(clip_model: str, clip_pretrained: str):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip  # noqa: PLC0415

        logger.info("Loading CLIP model: %s/%s", clip_model, clip_pretrained)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=device
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess


class EmbeddingService:
    """
    Generates text and image embeddings and stores them in Qdrant.
    Models are lazy-loaded as module-level singletons to avoid reloading.
    """

    def __init__(self) -> None:
        from app.config import get_settings  # noqa: PLC0415
        from app.services.qdrant_service import get_qdrant_client, ensure_collections  # noqa: PLC0415

        self._settings = get_settings()
        self._qdrant = get_qdrant_client()
        ensure_collections()

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a text embedding vector.

        Args:
            text: Input string.

        Returns:
            List[float] of length 384 (all-MiniLM-L6-v2).
        """
        model = _load_text_model(self._settings.text_embedding_model)
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Generate a CLIP image embedding vector.

        Args:
            image: PIL Image.

        Returns:
            List[float] of length 512 (ViT-B/32).
        """
        model, preprocess = _load_clip(
            self._settings.clip_model, self._settings.clip_pretrained
        )
        device = next(model.parameters()).device

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().tolist()

    # ── Qdrant upsert ─────────────────────────────────────────────────────────

    def upsert_video_chunk(
        self,
        chunk_id: str,
        text_vector: List[float],
        image_vector: Optional[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Upsert a video chunk into the video Qdrant collection.

        Args:
            chunk_id: UUID string used as Qdrant point ID.
            text_vector: Text embedding.
            image_vector: CLIP image embedding (or None → zero vector).
            metadata: Payload dict (speaker, timestamps, video_id, …).
        """
        from qdrant_client.http.models import PointStruct  # noqa: PLC0415
        from app.services.qdrant_service import IMAGE_DIM  # noqa: PLC0415

        img_vec = image_vector if image_vector else [0.0] * IMAGE_DIM
        point = PointStruct(
            id=str(uuid.UUID(chunk_id)),
            vector={"text": text_vector, "image": img_vec},
            payload=metadata,
        )
        self._qdrant.upsert(
            collection_name=self._settings.qdrant_video_collection,
            points=[point],
        )

    def upsert_pdf_chunk(
        self,
        chunk_id: str,
        text_vector: List[float],
        image_vector: Optional[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Upsert a PDF chunk into the PDF Qdrant collection.

        Args:
            chunk_id: UUID string used as Qdrant point ID.
            text_vector: Text embedding.
            image_vector: CLIP image embedding (or None → zero vector).
            metadata: Payload dict (pdf_id, page_number, …).
        """
        from qdrant_client.http.models import PointStruct  # noqa: PLC0415
        from app.services.qdrant_service import IMAGE_DIM  # noqa: PLC0415

        img_vec = image_vector if image_vector else [0.0] * IMAGE_DIM
        point = PointStruct(
            id=str(uuid.UUID(chunk_id)),
            vector={"text": text_vector, "image": img_vec},
            payload=metadata,
        )
        self._qdrant.upsert(
            collection_name=self._settings.qdrant_pdf_collection,
            points=[point],
        )

    def merge_text_for_chunk(
        self,
        transcript: Optional[str],
        ocr: Optional[str],
        caption: Optional[str],
    ) -> str:
        """
        Merge available text fields into a single string for embedding.
        Weights caption > transcript > ocr via concatenation ordering.
        """
        parts = []
        if caption:
            parts.append(caption.strip())
        if transcript:
            parts.append(transcript.strip())
        if ocr:
            parts.append(ocr.strip())
        return " ".join(parts)
