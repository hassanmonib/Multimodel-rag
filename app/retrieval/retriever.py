"""
Retriever: executes vector search in Qdrant with optional metadata filters,
merges text and image results, and reranks by score.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from qdrant_client.http.models.models import Filter, FieldCondition, MatchValue
except ImportError:
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.config import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.models.query_models import ChunkType, ParsedQuery, RetrievedResult, SourceType
from app.services.qdrant_service import get_qdrant_client

logger = logging.getLogger(__name__)


class Retriever:
    """
    Performs semantic retrieval from Qdrant:
    1. Apply speaker / source type metadata filters
    2. Text vector search
    3. If visual intent → image vector search (CLIP)
    4. Merge and rerank by score
    5. Fetch associated metadata from Qdrant payload
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_svc = EmbeddingService()
        self._qdrant = get_qdrant_client()

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[RetrievedResult]:
        """
        Retrieve the top-K most relevant chunks for a parsed query.

        Args:
            parsed_query: Structured query from QueryParser.
            top_k: Maximum number of results to return.
            score_threshold: Minimum similarity score (0–1).

        Returns:
            Sorted list of RetrievedResult objects (best first).
        """
        text_vec = self._embedding_svc.embed_text(parsed_query.semantic_topic)
        image_vec: Optional[List[float]] = None
        if parsed_query.has_visual_intent:
            # Use the text query to generate a CLIP text embedding for cross-modal search
            image_vec = self._clip_text_embed(parsed_query.semantic_topic)

        qdrant_filter = self._build_filter(parsed_query)
        collections = self._target_collections(parsed_query.source_type)

        all_results: Dict[str, RetrievedResult] = {}  # chunk_id → best result

        for collection in collections:
            chunk_type = (
                ChunkType.VIDEO
                if collection == self._settings.qdrant_video_collection
                else ChunkType.PDF
            )

            # ── Text search ───────────────────────────────────────────────────
            text_hits = self._query_vectors(
                collection_name=collection,
                vector=text_vec,
                vector_name="text",
                query_filter=qdrant_filter,
                limit=top_k * 2,
                score_threshold=score_threshold,
            )
            for hit in text_hits:
                result = self._hit_to_result(hit, chunk_type)
                if result.chunk_id not in all_results or result.score > all_results[result.chunk_id].score:
                    all_results[result.chunk_id] = result

            # ── Image search (if visual intent) ───────────────────────────────
            if image_vec:
                image_hits = self._query_vectors(
                    collection_name=collection,
                    vector=image_vec,
                    vector_name="image",
                    query_filter=qdrant_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                for hit in image_hits:
                    result = self._hit_to_result(hit, chunk_type)
                    if result.chunk_id not in all_results:
                        all_results[result.chunk_id] = result
                    else:
                        # Boost score if matched by both text and image
                        merged_score = min(
                            1.0,
                            (all_results[result.chunk_id].score + result.score) / 2
                            + 0.05,
                        )
                        all_results[result.chunk_id] = all_results[result.chunk_id].model_copy(
                            update={"score": merged_score}
                        )

        # ── Rerank and limit ──────────────────────────────────────────────────
        ranked = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        logger.info(
            "Retrieval: %d total candidates → top %d returned", len(ranked), top_k
        )
        return ranked[:top_k]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _target_collections(self, source_type: SourceType) -> List[str]:
        """Return Qdrant collection names to search based on source type."""
        video_col = self._settings.qdrant_video_collection
        pdf_col = self._settings.qdrant_pdf_collection
        if source_type == SourceType.VIDEO:
            return [video_col]
        if source_type == SourceType.PDF:
            return [pdf_col]
        return [video_col, pdf_col]

    def _query_vectors(
        self,
        collection_name: str,
        vector: List[float],
        vector_name: str,
        query_filter: Optional[Filter],
        limit: int,
        score_threshold: float,
    ) -> List[Any]:
        """
        Run vector search. Uses query_points (new API) or search (legacy) depending on client.
        Returns list of hits with .id, .score, .payload.
        """
        if hasattr(self._qdrant, "query_points"):
            response = self._qdrant.query_points(
                collection_name=collection_name,
                query=vector,
                using=vector_name,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True,
            )
            # QueryResponse has .points (list of ScoredPoint with id, score, payload)
            points = getattr(response, "points", None)
            return list(points) if points else []
        # Legacy client with .search()
        return self._qdrant.search(
            collection_name=collection_name,
            query_vector=(vector_name, vector),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

    def _build_filter(self, parsed_query: ParsedQuery) -> Optional[Filter]:
        """Construct a Qdrant metadata filter from parsed query fields."""
        conditions = []
        if parsed_query.speaker_name:
            conditions.append(
                FieldCondition(
                    key="speaker_name",
                    match=MatchValue(value=parsed_query.speaker_name),
                )
            )
        if not conditions:
            return None
        return Filter(must=conditions)

    def _clip_text_embed(self, text: str) -> List[float]:
        """
        Generate a CLIP text embedding for cross-modal image retrieval.
        Falls back to zero vector on failure.
        """
        try:
            import torch  # noqa: PLC0415
            import open_clip  # noqa: PLC0415

            # Reuse the cached CLIP model from embedding_service
            from app.embeddings.embedding_service import _load_clip, _clip_model  # noqa: PLC0415

            model, _ = _load_clip(
                self._settings.clip_model, self._settings.clip_pretrained
            )
            device = next(model.parameters()).device
            tokenizer = open_clip.get_tokenizer(self._settings.clip_model)
            tokens = tokenizer([text]).to(device)
            with torch.no_grad():
                features = model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
            return features.squeeze().cpu().tolist()
        except Exception as exc:
            logger.warning("CLIP text embedding failed: %s", exc)
            from app.services.qdrant_service import IMAGE_DIM  # noqa: PLC0415

            return [0.0] * IMAGE_DIM

    @staticmethod
    def _hit_to_result(hit: Any, chunk_type: ChunkType) -> RetrievedResult:
        """Convert a Qdrant ScoredPoint into a RetrievedResult."""
        payload: Dict[str, Any] = hit.payload or {}
        score = float(hit.score)
        chunk_id = str(hit.id)

        return RetrievedResult(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            score=score,
            # Video
            video_id=payload.get("video_id"),
            time_start=payload.get("time_start"),
            time_end=payload.get("time_end"),
            speaker_name=payload.get("speaker_name"),
            transcript_text=payload.get("transcript_text"),
            youtube_url=payload.get("youtube_url"),
            # PDF
            pdf_id=payload.get("pdf_id"),
            page_number=payload.get("page_number"),
            text_layer=payload.get("text_layer"),
            # Shared
            ocr_text=payload.get("ocr_text"),
            caption_text=payload.get("caption_text"),
            image_path=payload.get("image_path"),
        )
