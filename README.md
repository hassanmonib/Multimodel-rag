# Multimodal RAG Search

A production-ready Streamlit application for natural-language search inside **YouTube videos** and **PDF slide decks** — with speaker filtering, timestamp jumping, slide previews, and RAG-generated answers.

## Architecture

```
app/
├── app.py                   # Streamlit UI
├── config.py                # Pydantic Settings (env vars)
├── logging_config.py        # Structured logging
├── ingestion/
│   ├── youtube_ingestor.py  # YouTube pipeline (Whisper + diarization + CLIP)
│   └── pdf_ingestor.py      # PDF pipeline (PyMuPDF + OCR + BLIP + CLIP)
├── processing/
│   ├── ocr_processor.py     # pytesseract wrapper
│   ├── caption_generator.py # BLIP image captioning
│   ├── keyframe_extractor.py# OpenCV slide-change detector
│   └── diarization_processor.py  # pyannote.audio
├── embeddings/
│   └── embedding_service.py # sentence-transformers + open-clip + Qdrant upsert
├── retrieval/
│   ├── query_parser.py      # Speaker / intent / source extraction
│   └── retriever.py         # Qdrant dual-vector search + reranking
├── rag/
│   └── rag_engine.py        # Prompt builder + OpenAI LLM + citation output
├── services/
│   ├── qdrant_service.py    # Qdrant client + collection management
│   └── youtube_service.py   # yt-dlp helpers
├── database/
│   ├── connection.py        # SQLAlchemy engine + session
│   └── repositories.py      # CRUD: VideoRepository, PDFRepository, ChunkRepository
├── models/
│   ├── chunk_models.py      # VideoChunk, PDFChunk (Pydantic)
│   ├── db_models.py         # Video, PDF, Chunk (SQLAlchemy ORM)
│   └── query_models.py      # ParsedQuery, RetrievedResult, RAGAnswer (Pydantic)
└── utils/
    ├── text_utils.py        # highlight, format_timestamp, merge_text_fields
    └── image_utils.py       # save/load/resize/base64
```

## Prerequisites

| Service | How to run |
|---|---|
| **PostgreSQL** | Local install or Docker: `docker run -p 5432:5432 -e POSTGRES_PASSWORD=pass postgres` |
| **Qdrant** | `docker run -p 6333:6333 qdrant/qdrant` |
| **Tesseract OCR** | Windows: [download installer](https://github.com/UB-Mannheim/tesseract/wiki) |
| **ffmpeg** | Required by yt-dlp: `winget install ffmpeg` |

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env with your API keys and DB URLs

# 4. Run the app
streamlit run app/app.py
```

## Key environment variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | SQLAlchemy DB URL (PostgreSQL or SQLite) |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant server location |
| `OPENAI_API_KEY` | For RAG answer generation |
| `OPENAI_MODEL` | LLM model name (default: `gpt-4o-mini`) |
| `HUGGINGFACE_TOKEN` | For pyannote.audio speaker diarization (only if not using Deepgram) |
| `DEEPGRAM_API_KEY` | **Optional.** If set, use Deepgram for transcription + diarization instead of Whisper + pyannote (no HuggingFace token needed) |
| `DEEPGRAM_MODEL` | Deepgram model when using API (default: `nova-2`) |
| `TESSERACT_CMD` | Path to tesseract binary (Windows) |
| `HF_HUB_OFFLINE` | Set to `1` to use only cached Hugging Face models (no network). Run once with internet first to cache models. |

**Internet / offline:** The first run downloads embedding and optional caption models from Hugging Face. If you see a connection error, check your internet connection or firewall (access to `huggingface.co`). For offline use, run the app once with internet so models are cached, then set `HF_HUB_OFFLINE=1` in `.env`.

## Features

- 🎥 **YouTube ingestion** — transcript (Whisper), speaker diarization (pyannote), keyframe extraction, OCR, BLIP captioning
- 📄 **PDF ingestion** — page rendering (PyMuPDF), text extraction, OCR, BLIP captioning
- 🔍 **Semantic search** — sentence-transformers text + CLIP image embeddings in Qdrant
- 🎤 **Speaker filter** — query & sidebar filter by speaker name
- 🤖 **RAG answers** — OpenAI LLM generates cited answers from retrieved context
- ⏱ **Timestamp jump** — click to play video at exact moment
- 🖼 **Slide preview** — expand to view full PDF page image
