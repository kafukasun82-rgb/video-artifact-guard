"""Dark-frame and sudden underexposure drop detection."""

from __future__ import annotations

import numpy as np

from filters.base import BaseFilter, FilterResult
from filters.flicker import to_luma


class DarkFrameFilter(BaseFilter):
    """Flags crushed-black frames and abrupt luminance drops relative to clip baseline."""

    name = "dark_frame"

    def __init__(
        self,
        min_luma: float = 0.06,
        drop_ratio: float = 0.45,
        crushed_pixel_limit: float = 0.85,
    ) -> None:
        self.min_luma = min_luma
        self.drop_ratio = drop_ratio
        self.crushed_pixel_limit = crushed_pixel_limit

    def evaluate(self, frames: np.ndarray, fps: float = 24.0) -> FilterResult:
        if len(frames) == 0:
            return FilterResult(
                filter_name=self.name,
                passed=True,
                score=0.0,
                threshold=self.min_luma,
            )

        lumas = [to_luma(f) for f in frames]
        means = np.array([float(np.mean(lu)) for lu in lumas], dtype=np.float32)
        crushed_fracs = np.array(
            [float(np.mean(lu < 0.03)) for lu in lumas], dtype=np.float32
        )
        baseline = float(np.median(means))

        flagged: list[int] = []
        max_severity = 0.0
        for idx, (m, cf) in enumerate(zip(means, crushed_fracs)):
            is_too_dark = m < self.min_luma or cf > self.crushed_pixel_limit
            is_sudden_drop = baseline > 0.12 and (m < baseline * (1.0 - self.drop_ratio))
            if is_too_dark or is_sudden_drop:
                flagged.append(idx)
            severity = max(0.0, (baseline - float(m)) / max(baseline, 1e-4), float(cf))
            if severity > max_severity:
                max_severity = severity

        return FilterResult(
            filter_name=self.name,
            passed=len(flagged) == 0,
            score=round(max_severity, 6),
            threshold=self.drop_ratio,
            flagged_frames=flagged,
            metrics={
                "fps": fps,
                "median_luma": round(baseline, 4),
                "min_frame_luma": round(float(np.min(means)), 4),
                "max_crushed_fraction": round(float(np.max(crushed_fracs)), 4),
            },
        )
