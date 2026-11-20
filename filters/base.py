"""Base abstractions for video artifact filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FilterResult:
    """Structured output returned by a single artifact filter."""

    filter_name: str
    passed: bool
    score: float
    threshold: float
    flagged_frames: list[int] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_name": self.filter_name,
            "passed": bool(self.passed),
            "score": round(float(self.score), 6),
            "threshold": round(float(self.threshold), 6),
            "flagged_frames": [int(i) for i in self.flagged_frames],
            "metrics": self.metrics,
        }


class BaseFilter(ABC):
    """Abstract base class for pluggable CV quality-control filters."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, frames: np.ndarray, fps: float = 24.0) -> FilterResult:
        """Evaluate a sequence of video frames of shape (T, H, W, C) or (T, H, W)."""
