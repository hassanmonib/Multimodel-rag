"""
Deepgram transcription with speaker diarization.
Alternative to Whisper + pyannote: one API call returns transcript + speaker labels.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def transcribe_with_diarization(
    audio_path: str,
    api_key: str,
    *,
    model: str = "nova-2",
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float, str]]]:
    """
    Transcribe audio with Deepgram and return segments + diarization in the same
    format as Whisper + pyannote (so the rest of the pipeline is unchanged).

    Args:
        audio_path: Path to WAV/MP3 or other supported audio file.
        api_key: Deepgram API key.
        model: Deepgram model name (default nova-2).

    Returns:
        (transcript_segments, diarization_segs)
        - transcript_segments: list of {"text", "start", "end"} (like Whisper output).
        - diarization_segs: list of (start_sec, end_sec, "SPEAKER_N") for assign_speakers.
    """
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
    except ImportError:
        logger.warning("deepgram-sdk not installed. Install with: pip install deepgram-sdk")
        return [], []

    segments: List[Dict[str, Any]] = []
    diarization_segs: List[Tuple[float, float, str]] = []

    try:
        client = DeepgramClient(api_key=api_key)
        with open(audio_path, "rb") as f:
            payload = f.read()

        options = PrerecordedOptions(
            model=model,
            diarize=True,
            utterances=True,
            punctuate=True,
        )
        response = client.listen.rest.v("1").transcribe_file(payload, options)
    except Exception as exc:
        logger.error("Deepgram transcription failed: %s", exc)
        return [], []

    # Response: results.utterances (or results.channels[0].alternatives[0].utterances)
    result = response.to_dict() if hasattr(response, "to_dict") else response
    results_obj = result.get("results", {})
    utterances = results_obj.get("utterances", [])
    if not utterances and results_obj.get("channels"):
        alt = results_obj["channels"][0].get("alternatives", [{}])
        utterances = alt[0].get("utterances", []) if alt else []
    if not utterances:
        logger.warning("Deepgram returned no utterances")
        return [], []

    for u in utterances:
        start = float(u.get("start", 0))
        end = float(u.get("end", 0))
        transcript = (u.get("transcript") or "").strip()
        speaker = u.get("speaker", 0)
        speaker_label = f"SPEAKER_{speaker}"

        segments.append({"text": transcript, "start": start, "end": end})
        diarization_segs.append((start, end, speaker_label))

    logger.info(
        "Deepgram: %d utterance segments, %d diarization segments",
        len(segments),
        len(diarization_segs),
    )
    return segments, diarization_segs
