"""
RAG engine: constructs a prompt from retrieved chunks, calls the LLM,
and returns a structured RAGAnswer with citations.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI

from app.config import get_settings
from app.models.query_models import ChunkType, Citation, RAGAnswer, RetrievedResult
from app.utils.text_utils import format_timestamp

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful research assistant answering questions about video transcripts and PDF documents.

Rules:
1. Answer ONLY using the provided source snippets. Do NOT add information not present in the context.
2. Always cite your sources using the citation numbers [1], [2], etc. provided in the context.
3. Keep your answer concise, factual, and well-structured.
4. If the context doesn't contain enough information to answer, say "The provided sources don't contain enough information to answer this question."
"""

_USER_PROMPT_TEMPLATE = """Question: {question}

Sources:
{sources}

Answer (cite sources using [N] notation):"""


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Builds a structured prompt from retrieved chunks and calls the LLM.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        client_kwargs: dict = {"api_key": self._settings.openai_api_key}
        if self._settings.openai_base_url:
            client_kwargs["base_url"] = self._settings.openai_base_url
        self._client = OpenAI(**client_kwargs)

    def generate(
        self,
        question: str,
        results: List[RetrievedResult],
    ) -> RAGAnswer:
        """
        Generate a RAG answer for the given question using retrieved chunks.

        Args:
            question: Original user question.
            results: Top-K retrieved chunks.

        Returns:
            RAGAnswer with answer text and citations list.
        """
        if not results:
            return RAGAnswer(
                answer="No relevant sources found. Please try a different query or ingest more content.",
                citations=[],
            )

        sources_text, citations = self._build_sources(results)
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            question=question,
            sources=sources_text,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            answer_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            answer_text = f"Could not generate an answer (LLM error: {exc}). See source snippets below."

        return RAGAnswer(answer=answer_text.strip(), citations=citations)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_sources(
        results: List[RetrievedResult],
    ) -> tuple[str, List[Citation]]:
        """
        Format retrieved results into a numbered source list for the prompt
        and build the corresponding Citation objects.

        Returns:
            (sources_text, citations)
        """
        lines: List[str] = []
        citations: List[Citation] = []

        for idx, result in enumerate(results, start=1):
            if result.chunk_type == ChunkType.VIDEO:
                ts_start = format_timestamp(result.time_start or 0.0)
                ts_end = format_timestamp(result.time_end or 0.0)
                speaker = result.speaker_name or "Unknown Speaker"
                snippet = result.transcript_text or result.ocr_text or result.caption_text or ""
                label = f"{ts_start}–{ts_end} | {speaker}"
                source_header = f"[{idx}] VIDEO – {label}"
            else:
                page = result.page_number or "?"
                snippet = result.text_layer or result.ocr_text or result.caption_text or ""
                label = f"Page {page}"
                source_header = f"[{idx}] PDF – Page {page}"

            # Truncate snippet for prompt budget
            if len(snippet) > 600:
                snippet = snippet[:600] + " …"

            lines.append(f"{source_header}\n{snippet}")
            citations.append(
                Citation(
                    chunk_id=result.chunk_id,
                    chunk_type=result.chunk_type,
                    label=label,
                )
            )

        return "\n\n".join(lines), citations
