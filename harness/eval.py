"""Regression evaluation harness for video-artifact-guard."""

from __future__ import annotations

import json
from typing import Any

from filters.dark_frame import DarkFrameFilter
from filters.flicker import FlickerFilter
from harness.synthetic import (
    make_clean_pan_clip,
    make_dark_drop_clip,
    make_flicker_clip,
)
from pipeline.runner import ArtifactPipeline


def run_benchmark() -> dict[str, Any]:
    """Run the default pipeline across synthetic clean and defective clips."""
    pipe = ArtifactPipeline(filters=[FlickerFilter(), DarkFrameFilter()])
    cases = [
        ("clean_pan_slow", make_clean_pan_clip(pan_px_per_frame=1.5), True),
        ("clean_pan_fast", make_clean_pan_clip(pan_px_per_frame=3.5), True),
        ("flicker_strobe", make_flicker_clip(hf_amp=48.0, luma_amp=26.0), False),
        ("flicker_texture_only", make_flicker_clip(hf_amp=55.0, luma_amp=4.0), False),
        ("dark_frame_drop", make_dark_drop_clip(drop_frame=7), False),
    ]

    records: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0
    for name, clip, expected_pass in cases:
        report = pipe.run_frames(clip, fps=24.0, clip_id=name)
        matched = report.passed == expected_pass
        # Positive class = defect detected (passed == False)
        if not expected_pass and not report.passed:
            tp += 1
        elif expected_pass and not report.passed:
            fp += 1
        elif expected_pass and report.passed:
            tn += 1
        else:
            fn += 1

        records.append(
            {
                "case": name,
                "expected_pass": expected_pass,
                "actual_pass": report.passed,
                "matched": matched,
                "failed_filters": report.failed_filters,
                "elapsed_ms": round(report.elapsed_ms, 2),
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "total_cases": len(cases),
        "all_matched": all(r["matched"] for r in records),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cases": records,
    }


def main() -> None:
    summary = run_benchmark()
    print(json.dumps(summary, indent=2))
    if not summary["all_matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
