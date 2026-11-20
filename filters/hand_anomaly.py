"""Hand/extremity structural anomaly filter (heuristic CV baseline + pluggable landmark hook)."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from filters.base import BaseFilter, FilterResult

LandmarkScorer = Callable[[np.ndarray], float]


def _skin_region_instability(frames: np.ndarray) -> list[float]:
    """Lightweight fallback measuring abrupt frame-to-frame area/perimeter jitter in skin-tone blobs."""
    if len(frames) < 2:
        return [0.0] * len(frames)

    areas: list[float] = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            areas.append(0.0)
            continue
        if float(np.max(arr)) <= 1.5:
            arr = arr * 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        # Simple RGB skin-tone bounding heuristic
        skin_mask = (r > 95) & (g > 40) & (b > 20) & ((r - g) > 15) & (r > b)
        areas.append(float(np.mean(skin_mask)))

    scores = [0.0]
    for i in range(1, len(areas)):
        denom = max(areas[i - 1], areas[i], 0.01)
        scores.append(abs(areas[i] - areas[i - 1]) / denom)
    return scores


class HandAnomalyFilter(BaseFilter):
    """Detects malformed hand/extremity geometry via a pluggable keypoint scorer or CV fallback."""

    name = "hand_anomaly"

    def __init__(
        self,
        threshold: float = 0.65,
        landmark_scorer: Optional[LandmarkScorer] = None,
    ) -> None:
        self.threshold = threshold
        self.landmark_scorer = landmark_scorer

    def evaluate(self, frames: np.ndarray, fps: float = 24.0) -> FilterResult:
        if len(frames) == 0:
            return FilterResult(
                filter_name=self.name,
                passed=True,
                score=0.0,
                threshold=self.threshold,
            )

        if self.landmark_scorer is not None:
            scores = [float(self.landmark_scorer(f)) for f in frames]
            mode = "landmark_hook"
        else:
            scores = _skin_region_instability(frames)
            mode = "heuristic_skin_stability"

        flagged = [i for i, s in enumerate(scores) if s > self.threshold]
        peak = float(max(scores)) if scores else 0.0
        return FilterResult(
            filter_name=self.name,
            passed=len(flagged) == 0,
            score=round(peak, 6),
            threshold=self.threshold,
            flagged_frames=flagged,
            metrics={
                "fps": fps,
                "mode": mode,
                "mean_score": round(float(np.mean(scores)), 6),
            },
        )
