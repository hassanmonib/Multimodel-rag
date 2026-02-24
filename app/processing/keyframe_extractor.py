"""
Keyframe extraction using OpenCV frame-differencing to detect slide changes.
Returns a list of (timestamp_seconds, PIL.Image) pairs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """
    Detects visually distinct keyframes (slide changes) in a video file
    using frame-differencing and a configurable sensitivity threshold.
    """

    def __init__(
        self,
        diff_threshold: float = 30.0,
        min_interval_secs: float = 5.0,
        resize_width: int = 640,
    ) -> None:
        """
        Args:
            diff_threshold: Mean absolute pixel difference to count as a scene change.
            min_interval_secs: Minimum seconds between accepted keyframes.
            resize_width: Width to resize frames to before comparison (speeds up diffs).
        """
        self._diff_threshold = diff_threshold
        self._min_interval = min_interval_secs
        self._resize_width = resize_width

    def extract(self, video_path: str) -> List[Tuple[float, Image.Image]]:
        """
        Extract keyframes from a video file.

        Args:
            video_path: Absolute path to the video file.

        Returns:
            List of (timestamp_seconds, PIL.Image) tuples, sorted by timestamp.

        Raises:
            FileNotFoundError: If video_path does not exist.
            RuntimeError: If OpenCV cannot open the video.
        """
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        keyframes: List[Tuple[float, Image.Image]] = []
        prev_gray: np.ndarray | None = None
        last_keyframe_time: float = -self._min_interval

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                timestamp = frame_idx / fps

                # Resize for fast comparison
                h, w = frame.shape[:2]
                new_w = self._resize_width
                new_h = int(h * new_w / w)
                small = cv2.resize(frame, (new_w, new_h))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
                    is_scene_change = diff > self._diff_threshold
                    elapsed = timestamp - last_keyframe_time
                    if is_scene_change and elapsed >= self._min_interval:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        keyframes.append((timestamp, pil_img))
                        last_keyframe_time = timestamp
                        logger.debug("Keyframe at %.2fs (diff=%.2f)", timestamp, diff)
                else:
                    # Always include the very first frame
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    keyframes.append((0.0, pil_img))
                    last_keyframe_time = 0.0

                prev_gray = gray
        finally:
            cap.release()

        logger.info("Extracted %d keyframes from %s", len(keyframes), video_path)
        return keyframes
