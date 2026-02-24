"""
Speaker diarization using pyannote.audio.
Maps diarization segments onto Whisper transcript chunks.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DiarizationProcessor:
    """
    Runs speaker diarization on an audio file using pyannote.audio and
    assigns speaker labels to transcript time windows.
    """

    def __init__(self, hf_token: Optional[str] = None) -> None:
        """
        Args:
            hf_token: HuggingFace access token required to download pyannote models.
                      Falls back to config value if not supplied.
        """
        if hf_token is None:
            from app.config import get_settings  # noqa: PLC0415

            hf_token = get_settings().huggingface_token

        self._hf_token = hf_token
        self._pipeline = None  # lazy-loaded on first use

    def _load_pipeline(self) -> None:
        """Lazy-load the pyannote diarization pipeline."""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline  # noqa: PLC0415
            import torch  # noqa: PLC0415

            logger.info("Loading pyannote speaker diarization pipeline …")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = self._pipeline.to(device)
            logger.info("Diarization pipeline loaded on %s", device)
        except ImportError:
            logger.warning(
                "pyannote.audio not installed. Speaker diarization disabled."
            )
        except Exception as exc:
            logger.warning("Could not load diarization pipeline: %s", exc)

    def diarize(self, audio_path: str) -> List[Tuple[float, float, str]]:
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to a WAV or compatible audio file.

        Returns:
            List of (start_sec, end_sec, speaker_label) tuples, e.g.
            [(0.0, 5.3, 'SPEAKER_00'), (5.3, 12.1, 'SPEAKER_01'), ...]
            Returns empty list if diarization is unavailable or fails.
        """
        self._load_pipeline()
        if self._pipeline is None:
            return []

        try:
            diarization = self._pipeline(audio_path)
            segments: List[Tuple[float, float, str]] = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((segment.start, segment.end, speaker))
            logger.info("Diarization produced %d speaker segments", len(segments))
            return segments
        except Exception as exc:
            logger.error("Diarization failed: %s", exc)
            return []

    def assign_speakers(
        self,
        chunks: List[Dict],
        diarization_segments: List[Tuple[float, float, str]],
    ) -> List[Dict]:
        """
        Assign the dominant speaker label to each transcript chunk via
        majority-overlap voting.

        Args:
            chunks: List of dicts with at least 'time_start' and 'time_end'.
            diarization_segments: Output of self.diarize().

        Returns:
            The same list with a 'speaker_label' key added/updated on each chunk.
        """
        for chunk in chunks:
            start = chunk["time_start"]
            end = chunk["time_end"]
            overlap: Dict[str, float] = {}

            for seg_start, seg_end, speaker in diarization_segments:
                # Compute overlap between chunk window and diarization segment
                ov = max(0.0, min(end, seg_end) - max(start, seg_start))
                if ov > 0:
                    overlap[speaker] = overlap.get(speaker, 0.0) + ov

            if overlap:
                chunk["speaker_label"] = max(overlap, key=overlap.__getitem__)
            else:
                chunk["speaker_label"] = None

        return chunks
