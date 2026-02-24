"""
Multimodal RAG Search – Streamlit Application
Main entry point: run with `streamlit run app/app.py`
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

from app.config import get_settings
from app.database.connection import init_db
from app.database.repositories import ChunkRepository, PDFRepository, VideoRepository
from app.database.connection import get_session
from app.logging_config import setup_logging
from app.models.query_models import ChunkType, RetrievedResult, SourceType
from app.utils.image_utils import load_image, pil_to_bytes
from app.utils.text_utils import format_timestamp, highlight_query_words, extract_snippet

# ── App-level setup ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()
logger = logging.getLogger(__name__)

# ── Cached resource initialisation ───────────────────────────────────────────


@st.cache_resource(show_spinner="Initialising database …")
def _init_db():
    try:
        init_db()
        return True
    except Exception as exc:
        logger.error("DB init failed: %s", exc)
        return False


@st.cache_resource(show_spinner="Loading retrieval pipeline …")
def _get_retriever():
    from app.retrieval.retriever import Retriever  # noqa: PLC0415

    return Retriever()


@st.cache_resource(show_spinner="Loading query parser …")
def _get_query_parser():
    from app.retrieval.query_parser import QueryParser  # noqa: PLC0415

    return QueryParser()


@st.cache_resource(show_spinner="Loading RAG engine …")
def _get_rag_engine():
    from app.rag.rag_engine import RAGEngine  # noqa: PLC0415

    return RAGEngine()


def _get_youtube_ingestor():
    from app.ingestion.youtube_ingestor import YouTubeIngestor  # noqa: PLC0415

    return YouTubeIngestor()


def _get_pdf_ingestor():
    from app.ingestion.pdf_ingestor import PDFIngestor  # noqa: PLC0415

    return PDFIngestor()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _list_speakers() -> List[str]:
    """Fetch distinct speaker names from the database."""
    try:
        with get_session() as session:
            return ChunkRepository(session).list_speakers()
    except Exception:
        return []


def _list_videos():
    try:
        with get_session() as session:
            return VideoRepository(session).list_all()
    except Exception:
        return []


def _list_pdfs():
    try:
        with get_session() as session:
            return PDFRepository(session).list_all()
    except Exception:
        return []


def _render_image(image_path: Optional[str], caption: str = "", width: int = 280) -> None:
    """Safely render an image from disk in Streamlit."""
    if not image_path:
        return
    img = load_image(image_path)
    if img is None:
        return
    st.image(pil_to_bytes(img), caption=caption, width=width)


def _display_video_result(result: RetrievedResult, query: str) -> None:
    """Render a single video chunk result card."""
    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])

        with col_img:
            _render_image(result.image_path, caption="Keyframe")

        with col_info:
            ts_start = format_timestamp(result.time_start or 0.0)
            ts_end = format_timestamp(result.time_end or 0.0)
            speaker = result.speaker_name or "Unknown Speaker"

            st.markdown(f"**🎥 Video** &nbsp;|&nbsp; ⏱ `{ts_start} – {ts_end}` &nbsp;|&nbsp; 🎤 **{speaker}**")
            st.markdown(f"**Confidence:** `{result.score:.2%}`")

            snippet_text = result.transcript_text or result.ocr_text or result.caption_text or ""
            snippet = extract_snippet(snippet_text, 250)
            highlighted = highlight_query_words(snippet, query)
            st.markdown(highlighted)

            with st.expander("▶ Watch at this timestamp"):
                if result.youtube_url:
                    st.video(result.youtube_url, start_time=int(result.time_start or 0))
                else:
                    st.info("YouTube URL not available for this chunk.")


def _display_pdf_result(result: RetrievedResult, query: str) -> None:
    """Render a single PDF chunk result card."""
    with st.container(border=True):
        col_img, col_info = st.columns([1, 3])

        with col_img:
            _render_image(result.image_path, caption=f"Page {result.page_number}")

        with col_info:
            page = result.page_number or "?"
            st.markdown(f"**📄 PDF** &nbsp;|&nbsp; 📑 Page **{page}**")
            st.markdown(f"**Confidence:** `{result.score:.2%}`")

            snippet_text = result.text_layer or result.ocr_text or result.caption_text or ""
            snippet = extract_snippet(snippet_text, 250)
            highlighted = highlight_query_words(snippet, query)
            st.markdown(highlighted)

            with st.expander("🔍 View full slide"):
                if result.image_path and Path(result.image_path).is_file():
                    img = load_image(result.image_path)
                    if img:
                        st.image(pil_to_bytes(img), caption=f"Page {page}", use_container_width=True)
                else:
                    st.info("Slide image not available.")


def _display_rag_answer(rag_answer) -> None:
    """Display the RAG-generated answer with citation badges."""
    st.markdown("### 🤖 AI-Generated Answer")
    with st.container(border=True):
        st.markdown(rag_answer.answer)
        if rag_answer.citations:
            st.markdown("**Sources:**")
            citation_cols = st.columns(min(len(rag_answer.citations), 4))
            for i, citation in enumerate(rag_answer.citations):
                icon = "🎥" if citation.chunk_type == ChunkType.VIDEO else "📄"
                with citation_cols[i % 4]:
                    st.markdown(
                        f"<span style='background:#1e3a5f;padding:4px 8px;border-radius:12px;"
                        f"font-size:0.8em;color:#e0e8ff'>{icon} [{i+1}] {citation.label}</span>",
                        unsafe_allow_html=True,
                    )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────


def _render_sidebar() -> tuple[SourceType, Optional[str], int, float]:
    """Render sidebar filters. Returns (source_type, speaker, top_k, threshold)."""
    with st.sidebar:
        st.markdown("## ⚙️ Search Filters")
        st.divider()

        source_label = st.radio(
            "📂 Source",
            options=["Both", "YouTube Videos", "PDFs"],
            index=0,
            horizontal=False,
        )
        source_map = {
            "Both": SourceType.BOTH,
            "YouTube Videos": SourceType.VIDEO,
            "PDFs": SourceType.PDF,
        }
        source_type = source_map[source_label]

        st.divider()
        speakers = _list_speakers()
        speaker_options = ["All Speakers"] + speakers
        selected_speaker = st.selectbox("🎤 Filter by Speaker", speaker_options)
        speaker_filter: Optional[str] = None if selected_speaker == "All Speakers" else selected_speaker

        st.divider()
        top_k = st.slider("🎯 Top-K Results", min_value=1, max_value=20, value=5)
        threshold = st.slider(
            "📊 Min. Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05
        )

        st.divider()
        st.markdown("### 📚 Ingested Content")
        videos = _list_videos()
        pdfs = _list_pdfs()
        if videos:
            with st.expander(f"🎥 Videos ({len(videos)})"):
                for v in videos:
                    st.markdown(f"- [{v.title or v.id}]({v.youtube_url})")
        if pdfs:
            with st.expander(f"📄 PDFs ({len(pdfs)})"):
                for p in pdfs:
                    st.markdown(f"- {p.title or p.filename} ({p.total_pages} pages)")

    return source_type, speaker_filter, top_k, threshold


# ── INGEST TAB ────────────────────────────────────────────────────────────────


def _render_ingest_tab() -> None:
    """Render the data ingestion UI."""
    col_yt, col_pdf = st.columns(2)

    with col_yt:
        st.markdown("### 🎬 YouTube Video")
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key="yt_url_input",
        )
        if st.button("⬇️ Ingest YouTube Video", key="ingest_yt", use_container_width=True):
            if not youtube_url.strip():
                st.error("Please enter a valid YouTube URL.")
            else:
                progress_bar = st.progress(0, text="Starting ingestion …")
                status_area = st.empty()
                try:
                    ingestor = _get_youtube_ingestor()
                    for step, total, message in ingestor.ingest(youtube_url.strip()):
                        pct = int(step / total * 100)
                        progress_bar.progress(pct, text=message)
                        status_area.info(message)
                    st.success("✅ YouTube video ingested successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")
                    logger.exception("YouTube ingestion error")

    with col_pdf:
        st.markdown("### 📄 PDF Document")
        pdf_file = st.file_uploader(
            "Upload PDF Slide Deck",
            type=["pdf"],
            key="pdf_uploader",
        )
        if st.button("📤 Ingest PDF", key="ingest_pdf", use_container_width=True):
            if pdf_file is None:
                st.error("Please upload a PDF file first.")
            else:
                progress_bar = st.progress(0, text="Starting PDF ingestion …")
                status_area = st.empty()
                try:
                    pdf_bytes = pdf_file.read()
                    ingestor = _get_pdf_ingestor()
                    for step, total, message in ingestor.ingest(pdf_bytes, pdf_file.name):
                        pct = int(step / total * 100)
                        progress_bar.progress(pct, text=message)
                        status_area.info(message)
                    st.success(f"✅ '{pdf_file.name}' ingested successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ PDF ingestion failed: {exc}")
                    logger.exception("PDF ingestion error")


# ── SEARCH TAB ────────────────────────────────────────────────────────────────


def _render_search_tab(
    source_type: SourceType,
    speaker_filter: Optional[str],
    top_k: int,
    threshold: float,
) -> None:
    """Render the search UI with result display."""
    query = st.text_input(
        "🔍 Search inside videos and PDFs",
        placeholder='e.g. "Find where Ali talks about success in life"',
        key="search_query",
    )

    search_clicked = st.button("Search", type="primary", use_container_width=False, key="search_btn")

    if not search_clicked or not query.strip():
        st.markdown(
            """
            <div style="text-align:center;padding:60px 0;color:#666;">
                <h3>Enter a query above to search</h3>
                <p>Try: <em>"Find where Ali talks about success"</em> or <em>"Show charts about revenue growth"</em></p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    with st.spinner("🔎 Searching and generating answer …"):
        try:
            parser = _get_query_parser()
            retriever = _get_retriever()
            rag = _get_rag_engine()

            parsed = parser.parse(query.strip())

            # Override source type and speaker from sidebar if set
            if source_type != SourceType.BOTH:
                parsed = parsed.model_copy(update={"source_type": source_type})
            if speaker_filter:
                parsed = parsed.model_copy(update={"speaker_name": speaker_filter})

            results: List[RetrievedResult] = retriever.retrieve(
                parsed, top_k=top_k, score_threshold=threshold
            )

            rag_answer = rag.generate(question=query.strip(), results=results)

        except Exception as exc:
            st.error(f"❌ Search failed: {exc}")
            logger.exception("Search error")
            return

    # ── RAG Answer ────────────────────────────────────────────────────────────
    _display_rag_answer(rag_answer)

    st.divider()
    st.markdown(f"### 📋 Retrieved Sources ({len(results)} results)")

    if not results:
        st.info("No results found. Try a different query or lower the confidence threshold.")
        return

    # ── Result cards ──────────────────────────────────────────────────────────
    for result in results:
        if result.chunk_type == ChunkType.VIDEO:
            _display_video_result(result, query)
        else:
            _display_pdf_result(result, query)

        st.markdown("")  # spacing


# ── MAIN ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry point."""

    # Initialise core infrastructure
    db_ok = _init_db()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                    padding:32px 40px;border-radius:16px;margin-bottom:24px;">
            <h1 style="color:#e0e8ff;margin:0;font-size:2.4rem;">
                🔍 Multimodal RAG Search
            </h1>
            <p style="color:#8ab4f8;margin:8px 0 0;font-size:1.1rem;">
                Search inside YouTube videos &amp; PDF slide decks using natural language.
                Speaker filtering · Timestamp jumping · Slide previews · AI-generated answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not db_ok:
        st.error(
            "⚠️ Database connection failed. Check your `DATABASE_URL` in `.env` and ensure "
            "PostgreSQL (or SQLite) is accessible."
        )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    source_type, speaker_filter, top_k, threshold = _render_sidebar()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_search, tab_ingest = st.tabs(["🔍 Search", "📥 Ingest Content"])

    with tab_search:
        _render_search_tab(source_type, speaker_filter, top_k, threshold)

    with tab_ingest:
        _render_ingest_tab()


if __name__ == "__main__":
    main()
