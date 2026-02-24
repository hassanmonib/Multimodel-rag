"""
YouTube helper service using yt-dlp.
Provides video metadata fetching and audio/video downloading.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_video_metadata(url: str) -> Dict[str, Any]:
    """
    Fetch YouTube video metadata without downloading.

    Args:
        url: YouTube video URL.

    Returns:
        Dict with keys: id, title, channel, duration, thumbnail, webpage_url.

    Raises:
        RuntimeError: If yt-dlp fails to extract info.
    """
    import yt_dlp  # noqa: PLC0415

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise RuntimeError(f"yt-dlp metadata extraction failed: {exc}") from exc

    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "channel": info.get("channel") or info.get("uploader", ""),
        "duration": float(info.get("duration") or 0.0),
        "thumbnail": info.get("thumbnail", ""),
        "webpage_url": info.get("webpage_url", url),
    }


def download_video(
    url: str,
    output_dir: str,
    video_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Download a YouTube video and its audio stream to output_dir.

    Args:
        url: YouTube video URL.
        output_dir: Directory where files will be saved.
        video_id: Optional ID used for deterministic output filenames.

    Returns:
        Dict with keys 'video_path' and 'audio_path' (absolute paths).

    Raises:
        RuntimeError: If download fails.
    """
    import yt_dlp  # noqa: PLC0415

    os.makedirs(output_dir, exist_ok=True)
    base_name = video_id or "%(id)s"

    # ── Download video (best quality, max 1080p) ──────────────────────────────
    video_path = str(Path(output_dir) / f"{base_name}.mp4")
    ydl_video_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": video_path,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
    }

    # ── Download audio only (WAV for whisper) ─────────────────────────────────
    audio_path = str(Path(output_dir) / f"{base_name}.wav")
    ydl_audio_opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(Path(output_dir) / f"{base_name}_audio.%(ext)s"),
        "format": "bestaudio",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_video_opts) as ydl:
            ydl.download([url])
            # Resolve actual video output path
            info = ydl.extract_info(url, download=False)
            video_path = str(Path(output_dir) / f"{info['id']}.mp4")

        with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            audio_path = str(
                Path(output_dir) / f"{info['id']}_audio.wav"
            )
    except Exception as exc:
        raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

    logger.info("Downloaded video→%s  audio→%s", video_path, audio_path)
    return {"video_path": video_path, "audio_path": audio_path}
