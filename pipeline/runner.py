"""Composable artifact QC pipeline runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from filters.base import BaseFilter, FilterResult
from filters.dark_frame import DarkFrameFilter
from filters.flicker import FlickerFilter
from filters.hand_anomaly import HandAnomalyFilter
from pipeline.io import load_video_frames


@dataclass
class ClipReport:
    """Aggregated quality-control report for a single video clip."""

    clip_id: str
    passed: bool
    num_frames: int
    fps: float
    elapsed_ms: float
    results: list[FilterResult] = field(default_factory=list)

    @property
    def failed_filters(self) -> list[str]:
        return [r.filter_name for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "passed": bool(self.passed),
            "num_frames": int(self.num_frames),
            "fps": round(float(self.fps), 3),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "failed_filters": self.failed_filters,
            "results": [r.to_dict() for r in self.results],
        }


class ArtifactPipeline:
    """Runs an ordered sequence of CV artifact filters over decoded frames or video files."""

    def __init__(self, filters: Optional[Sequence[BaseFilter]] = None) -> None:
        if filters is None:
            self.filters: list[BaseFilter] = [
                FlickerFilter(),
                DarkFrameFilter(),
                HandAnomalyFilter(),
            ]
        else:
            self.filters = list(filters)

    def run_frames(
        self,
        frames: np.ndarray,
        fps: float = 24.0,
        clip_id: str = "memory_clip",
    ) -> ClipReport:
        t0 = time.perf_counter()
        results = [flt.evaluate(frames, fps=fps) for flt in self.filters]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = all(r.passed for r in results)
        return ClipReport(
            clip_id=clip_id,
            passed=passed,
            num_frames=len(frames),
            fps=fps,
            elapsed_ms=elapsed_ms,
            results=results,
        )

    def run_video(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        stride: int = 1,
        resize_wh: Optional[tuple[int, int]] = (320, 180),
    ) -> ClipReport:
        frames, eff_fps = load_video_frames(
            video_path, max_frames=max_frames, stride=stride, resize_wh=resize_wh
        )
        return self.run_frames(frames, fps=eff_fps, clip_id=str(video_path))
