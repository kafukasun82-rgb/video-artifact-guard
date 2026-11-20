"""Synthetic clip generators for deterministic filter regression testing."""

from __future__ import annotations

import numpy as np


def _base_scene(h: int, w: int, shift_x: float = 0.0) -> np.ndarray:
    """Create a smooth multi-scale synthetic RGB scene shifted horizontally by `shift_x` pixels."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_s = xx - shift_x
    # Smooth landscape-like gradients + moderate spatial texture
    r = 0.45 + 0.22 * np.sin(2.0 * np.pi * x_s / 64.0) * np.cos(2.0 * np.pi * yy / 48.0)
    g = 0.48 + 0.18 * np.cos(2.0 * np.pi * x_s / 40.0 + 0.5)
    b = 0.50 + 0.20 * np.sin(2.0 * np.pi * yy / 56.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def make_clean_pan_clip(
    n_frames: int = 16, h: int = 96, w: int = 128, pan_px_per_frame: float = 2.0
) -> np.ndarray:
    """Generate a clean clip with smooth horizontal camera panning."""
    return np.stack(
        [_base_scene(h, w, shift_x=i * pan_px_per_frame) for i in range(n_frames)],
        axis=0,
    )


def make_flicker_clip(
    n_frames: int = 16,
    h: int = 96,
    w: int = 128,
    hf_amp: float = 48.0,
    luma_amp: float = 26.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a clip with alternating high-frequency texture boiling and luminance strobing."""
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    # Checkerboard / fine high-frequency grid injected on alternating frames
    yy, xx = np.mgrid[0:h, 0:w]
    hf_pattern = (((yy + xx) % 2) * 2 - 1).astype(np.float32)[..., None]

    for i in range(n_frames):
        base = _base_scene(h, w, shift_x=i * 0.5).astype(np.float32)
        if i % 2 == 1:
            noise = rng.normal(0.0, hf_amp * 0.35, size=(h, w, 1)).astype(np.float32)
            base = base + hf_pattern * hf_amp + noise + luma_amp
        else:
            base = base - luma_amp * 0.5
        frames.append(np.clip(base, 0, 255).astype(np.uint8))
    return np.stack(frames, axis=0)


def make_dark_drop_clip(
    n_frames: int = 16, h: int = 96, w: int = 128, drop_frame: int = 7
) -> np.ndarray:
    """Generate a clean clip with a single sudden underexposed/crushed-black frame."""
    clip = make_clean_pan_clip(n_frames=n_frames, h=h, w=w, pan_px_per_frame=1.5).copy()
    clip[drop_frame] = (clip[drop_frame].astype(np.float32) * 0.05).astype(np.uint8)
    return clip
