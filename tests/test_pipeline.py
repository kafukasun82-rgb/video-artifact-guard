"""Integration tests for ArtifactPipeline and evaluation harness."""

from __future__ import annotations

from filters.dark_frame import DarkFrameFilter
from filters.flicker import FlickerFilter
from filters.hand_anomaly import HandAnomalyFilter
from harness.eval import run_benchmark
from harness.synthetic import make_clean_pan_clip, make_dark_drop_clip
from pipeline.runner import ArtifactPipeline


def test_dark_frame_filter_flags_exact_drop_frame() -> None:
    clip = make_dark_drop_clip(n_frames=12, drop_frame=5)
    res = DarkFrameFilter().evaluate(clip)
    assert res.passed is False
    assert 5 in res.flagged_frames


def test_pipeline_report_serialization() -> None:
    clip = make_clean_pan_clip(n_frames=10)
    pipe = ArtifactPipeline(
        filters=[FlickerFilter(), DarkFrameFilter(), HandAnomalyFilter()]
    )
    report = pipe.run_frames(clip, fps=24.0, clip_id="unit_clean")
    data = report.to_dict()

    assert data["passed"] is True
    assert data["clip_id"] == "unit_clean"
    assert data["failed_filters"] == []
    assert len(data["results"]) == 3


def test_synthetic_harness_all_matched() -> None:
    summary = run_benchmark()
    assert summary["all_matched"] is True
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
