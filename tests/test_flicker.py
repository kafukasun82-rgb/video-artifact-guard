"""Unit tests for filters/flicker.py."""

from __future__ import annotations

import numpy as np

from filters.flicker import (
    FlickerFilter,
    compute_hf_energy,
    compute_motion_compensated_flicker,
    estimate_motion_vectors,
    to_luma,
)
from harness.synthetic import make_clean_pan_clip, make_flicker_clip


def test_hf_energy_higher_for_checkerboard_than_smooth_frame() -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = 0.5 + 0.25 * np.sin(2.0 * np.pi * xx / 64.0)
    checker = ((yy + xx) % 2).astype(np.float32)

    assert compute_hf_energy(checker) > compute_hf_energy(smooth) * 5.0


def test_motion_vectors_detect_horizontal_pan_both_backends() -> None:
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    # Shift right by 2 pixels via circular roll (well-conditioned texture for optical flow)
    l0 = base
    l1 = np.roll(base, shift=2, axis=1)

    mv_cv2 = estimate_motion_vectors(l0, l1, use_cv2=True)
    mv_scipy = estimate_motion_vectors(l0, l1, use_cv2=False, block_size=16)

    assert mv_cv2.shape[-1] == 2
    assert mv_scipy.shape[-1] == 2
    # Interior crop avoids border roll wrap-around
    assert float(np.mean(mv_cv2[8:-8, 8:-8, 0])) > 1.0
    assert float(np.mean(mv_scipy[1:-1, 1:-1, 0])) > 1.0


def test_flicker_filter_separates_clean_pan_from_flicker() -> None:
    clean = make_clean_pan_clip(n_frames=12, pan_px_per_frame=2.5)
    noisy = make_flicker_clip(n_frames=12, hf_amp=48.0, luma_amp=26.0)

    flt = FlickerFilter()
    res_clean = flt.evaluate(clean)
    res_noisy = flt.evaluate(noisy)

    assert res_clean.passed is True
    assert res_noisy.passed is False
    assert res_noisy.score > res_clean.score * 3.0
    assert len(res_noisy.flagged_frames) > 0


def test_flicker_analysis_handles_short_clips() -> None:
    tiny = make_clean_pan_clip(n_frames=2)
    out = compute_motion_compensated_flicker(tiny)
    assert out["clip_score"] == 0.0
    assert len(out["hf_energies"]) == 2
