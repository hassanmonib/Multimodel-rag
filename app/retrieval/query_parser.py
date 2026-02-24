"""
Query parser: converts a raw natural-language query into a structured
ParsedQuery with extracted speaker name, visual intent flags, and source type.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.models.query_models import ParsedQuery, SourceType

logger = logging.getLogger(__name__)

# Keywords that suggest the user wants to see visual content (slides, graphs, etc.)
_VISUAL_KEYWORDS = frozenset(
    {
        "slide", "slides", "graph", "chart", "image", "picture", "photo",
        "diagram", "figure", "table", "visual", "show", "display", "screen",
        "presentation", "frame", "screenshot",
    }
)

# Keywords that indicate a YouTube / video source
_VIDEO_KEYWORDS = frozenset(
    {
        "video", "youtube", "talk", "lecture", "interview", "podcast",
        "clip", "watch", "says", "said", "mentioned", "spoke", "speaking",
        "timestamp",
    }
)

# Keywords that indicate a PDF / document source
_PDF_KEYWORDS = frozenset(
    {
        "pdf", "document", "paper", "slide deck", "presentation", "page",
        "chapter",
    }
)

# Common speaker introduction patterns:
#   "Find where Ali talks …", "What does Dr Smith say …", "When Maria discusses …"
_SPEAKER_PATTERNS = [
    re.compile(r"\bwhere\s+(?:does\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:talks?|says?|discusses?|mentions?|explains?|speaks?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:say|explain|mention|discuss)\b", re.IGNORECASE),
    re.compile(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
    re.compile(r"\bspeaker[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", re.IGNORECASE),
]


class QueryParser:
    """
    Lightweight heuristic query parser.

    Extracts:
    - speaker_name from common natural-language patterns
    - has_visual_intent from visual keyword detection
    - source_type from video/pdf keyword detection
    - semantic_topic as the cleaned query (stopwords / extracted parts removed)
    """

    def parse(self, raw_query: str) -> ParsedQuery:
        """
        Parse a raw natural-language query into a structured ParsedQuery.

        Args:
            raw_query: User's typed search string.

        Returns:
            ParsedQuery with extracted fields.
        """
        raw_query = raw_query.strip()
        speaker_name = self._extract_speaker(raw_query)
        has_visual = self._has_visual_intent(raw_query)
        source_type = self._detect_source(raw_query)
        semantic_topic = self._build_semantic_topic(raw_query, speaker_name)

        parsed = ParsedQuery(
            raw_query=raw_query,
            semantic_topic=semantic_topic,
            speaker_name=speaker_name,
            has_visual_intent=has_visual,
            source_type=source_type,
        )
        logger.debug("Parsed query: %s", parsed.model_dump())
        return parsed

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_speaker(query: str) -> Optional[str]:
        """Return the first matched speaker name, or None."""
        for pattern in _SPEAKER_PATTERNS:
            match = pattern.search(query)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name.lower() not in {"the", "a", "an", "this", "that"}:
                    return name
        return None

    @staticmethod
    def _has_visual_intent(query: str) -> bool:
        """Return True if any visual keyword appears in the query."""
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        return bool(tokens & _VISUAL_KEYWORDS)

    @staticmethod
    def _detect_source(query: str) -> SourceType:
        """Determine whether the query targets videos, PDFs, or both."""
        lower = query.lower()
        tokens = set(re.findall(r"\b\w+\b", lower))
        has_video = bool(tokens & _VIDEO_KEYWORDS)
        has_pdf = bool(tokens & _PDF_KEYWORDS)

        if has_video and not has_pdf:
            return SourceType.VIDEO
        if has_pdf and not has_video:
            return SourceType.PDF
        return SourceType.BOTH

    @staticmethod
    def _build_semantic_topic(query: str, speaker_name: Optional[str]) -> str:
        """
        Build a cleaned topic string for embedding by removing speaker
        attribution phrases from the query.
        """
        topic = query
        if speaker_name:
            # Remove patterns like "Find where Ali talks about" → keep the subject
            topic = re.sub(
                rf"\b(?:find\s+)?where\s+(?:does\s+)?{re.escape(speaker_name)}\s+(?:talks?|says?|discusses?|mentions?|explains?|speaks?)\s+(?:about\s+)?",
                "",
                topic,
                flags=re.IGNORECASE,
            )
            topic = re.sub(
                rf"\bby\s+{re.escape(speaker_name)}\b",
                "",
                topic,
                flags=re.IGNORECASE,
            )
        # Remove filler phrases
        topic = re.sub(
            r"\b(find|search|look for|show me|what is|when does|where does|tell me about)\b",
            "",
            topic,
            flags=re.IGNORECASE,
        )
        topic = re.sub(r"\s{2,}", " ", topic).strip()
        return topic or query
