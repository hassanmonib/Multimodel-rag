"""
Qdrant vector database client management.
Handles client initialisation, collection creation with dual named vectors,
and health checking.
"""
from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient

from app.config import get_settings

logger = logging.getLogger(__name__)

# Embedding dimensions
TEXT_DIM = 384   # all-MiniLM-L6-v2
IMAGE_DIM = 512  # CLIP ViT-B/32

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return (or create) a cached Qdrant client."""
    global _client
    if _client is None:
        settings = get_settings()
        if settings.qdrant_url:
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            logger.info("Qdrant client connected to cloud: %s", settings.qdrant_url)
        else:
            _client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
            logger.info(
                "Qdrant client connected to %s:%d",
                settings.qdrant_host,
                settings.qdrant_port,
            )
    return _client


def _dual_vector_config() -> dict:
    """Return a vectors_config dict with 'text' and 'image' named vectors.
    Uses plain dicts to avoid importing from qdrant_client.http.models (avoids
    NamedVectorParams import errors in some qdrant-client versions).
    """
    return {
        "text": {"size": TEXT_DIM, "distance": "Cosine"},
        "image": {"size": IMAGE_DIM, "distance": "Cosine"},
    }


def ensure_collections() -> None:
    """
    Create Qdrant collections for video and PDF chunks if they don't exist.
    Each collection uses dual named vectors: 'text' and 'image'.
    """
    client = get_qdrant_client()
    settings = get_settings()

    existing = {c.name for c in client.get_collections().collections}

    for collection_name in (
        settings.qdrant_video_collection,
        settings.qdrant_pdf_collection,
    ):
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=_dual_vector_config(),
            )
            logger.info("Created Qdrant collection: %s", collection_name)
        else:
            logger.debug("Qdrant collection already exists: %s", collection_name)


def delete_video_chunks(video_id: str) -> None:
    """
    Delete all points in the video_chunks collection whose payload video_id matches.
    Use when re-ingesting a video so old vectors are removed before upserting new ones.
    """
    try:
        try:
            from qdrant_client.http.models.models import (
                FieldCondition,
                Filter,
                FilterSelector,
                MatchValue,
            )
        except ImportError:
            from qdrant_client.http.models import (
                FieldCondition,
                Filter,
                FilterSelector,
                MatchValue,
            )
    except ImportError:
        logger.warning("Could not import Qdrant filter models; skip deleting old video chunks")
        return

    client = get_qdrant_client()
    settings = get_settings()
    client.delete(
        collection_name=settings.qdrant_video_collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id),
                    ),
                ],
            ),
        ),
    )
    logger.info("Deleted Qdrant points for video_id=%s", video_id)


def check_health() -> bool:
    """Return True if Qdrant is reachable, False otherwise."""
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return False
