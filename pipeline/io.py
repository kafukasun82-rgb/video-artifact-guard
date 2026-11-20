"""Video frame decoding utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_video_frames(
    video_path: str | Path,
    max_frames: Optional[int] = None,
    stride: int = 1,
    resize_wh: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, float]:
    """Decode RGB frames from a video file using OpenCV.

    Returns:
        (frames, effective_fps) where `frames` has shape (T, H, W, 3) uint8 RGB.
    """
    import cv2  # type: ignore

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video stream: {path}")

    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    step = max(1, int(stride))
    frames: list[np.ndarray] = []
    idx = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if resize_wh is not None:
                    rgb = cv2.resize(rgb, resize_wh, interpolation=cv2.INTER_AREA)
                frames.append(rgb)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), raw_fps / step
    return np.stack(frames, axis=0), raw_fps / step
