"""Temporal flicker detection using high-frequency energy and motion vectors.

Combines radial 2D FFT high-frequency energy oscillation with dense/block
motion-vector estimation to separate genuine camera/subject movement from
AI-generated temporal strobing and texture boiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import fft2, fftshift, ifft2
from scipy.ndimage import laplace

from filters.base import BaseFilter, FilterResult

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def to_luma(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB/BGR or grayscale frame into float32 luma normalized to [0, 1]."""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        # Rec. 709 luma approximation (channels treated as RGB)
        luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    elif arr.ndim == 2:
        luma = arr
    else:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")

    if float(np.max(luma)) > 1.5:
        luma = luma / 255.0
    return np.clip(luma, 0.0, 1.0)


def compute_hf_energy(luma: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    """Compute the normalized high-frequency spectral energy ratio of a luma frame.

    Uses a radial mask in the shifted 2D FFT magnitude spectrum outside
    `cutoff_ratio * min(H, W) / 2` from DC.
    """
    h, w = luma.shape
    centered = luma - float(np.mean(luma))
    spectrum = np.abs(fftshift(fft2(centered))) ** 2
    total_energy = float(np.sum(spectrum))
    if total_energy <= 1e-12:
        return 0.0

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = cutoff_ratio * (min(h, w) / 2.0)
    hf_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) >= (radius**2)
    hf_energy = float(np.sum(spectrum[hf_mask]))
    return hf_energy / total_energy


def compute_laplacian_energy(luma: np.ndarray) -> float:
    """Compute spatial high-frequency energy via the discrete Laplacian variance."""
    return float(np.mean(laplace(luma) ** 2))


def _block_phase_correlation_flow(
    prev_luma: np.ndarray, curr_luma: np.ndarray, block_size: int = 16
) -> np.ndarray:
    """Pure NumPy/SciPy fallback estimating per-block (dx, dy) motion vectors via phase correlation."""
    h, w = prev_luma.shape
    rows = max(1, h // block_size)
    cols = max(1, w // block_size)
    flow = np.zeros((rows, cols, 2), dtype=np.float32)

    for r in range(rows):
        y0, y1 = r * block_size, (r + 1) * block_size
        for c in range(cols):
            x0, x1 = c * block_size, (c + 1) * block_size
            b0 = prev_luma[y0:y1, x0:x1]
            b1 = curr_luma[y0:y1, x0:x1]
            f0 = fft2(b0 - float(np.mean(b0)))
            f1 = fft2(b1 - float(np.mean(b1)))
            cross_power = f0 * np.conj(f1)
            denom = np.abs(cross_power) + 1e-8
            corr = np.real(ifft2(cross_power / denom))
            peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)

            dy = float(peak_y - block_size if peak_y > block_size // 2 else peak_y)
            dx = float(peak_x - block_size if peak_x > block_size // 2 else peak_x)
            flow[r, c, 0] = -dx
            flow[r, c, 1] = -dy
    return flow


def estimate_motion_vectors(
    prev_luma: np.ndarray,
    curr_luma: np.ndarray,
    use_cv2: bool = True,
    block_size: int = 16,
) -> np.ndarray:
    """Estimate a 2-channel (dx, dy) motion-vector field between consecutive luma frames."""
    if use_cv2 and _HAS_CV2:
        prev_u8 = np.clip(prev_luma * 255.0, 0, 255).astype(np.uint8)
        curr_u8 = np.clip(curr_luma * 255.0, 0, 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev_u8,
            curr_u8,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow.astype(np.float32)
    return _block_phase_correlation_flow(prev_luma, curr_luma, block_size=block_size)


def compute_motion_compensated_flicker(
    frames: np.ndarray,
    cutoff_ratio: float = 0.25,
    motion_damping: float = 0.25,
    use_cv2: bool = True,
) -> dict[str, Any]:
    """Compute per-frame high-frequency energy, motion-vector stats, and flicker scores.

    Flicker manifests as second-order temporal oscillation in high-frequency
    energy and frame luminance that cannot be explained by coherent motion
    vectors. Smooth camera pans produce coherent, high-magnitude motion vectors
    and are damped by `motion_damping`.
    """
    n_frames = len(frames)
    if n_frames < 3:
        return {
            "hf_energies": [compute_hf_energy(to_luma(f), cutoff_ratio) for f in frames],
            "motion_magnitudes": [],
            "motion_coherence": [],
            "frame_flicker_scores": [0.0] * n_frames,
            "clip_score": 0.0,
        }

    lumas = [to_luma(f) for f in frames]
    hf_energies = np.array(
        [compute_hf_energy(lu, cutoff_ratio=cutoff_ratio) for lu in lumas],
        dtype=np.float32,
    )
    lap_energies = np.array(
        [compute_laplacian_energy(lu) for lu in lumas], dtype=np.float32
    )
    mean_lumas = np.array([float(np.mean(lu)) for lu in lumas], dtype=np.float32)

    motion_mags: list[float] = []
    motion_coherence: list[float] = []

    for i in range(n_frames - 1):
        mv = estimate_motion_vectors(lumas[i], lumas[i + 1], use_cv2=use_cv2)
        dx = mv[..., 0]
        dy = mv[..., 1]
        mag = np.sqrt(dx**2 + dy**2)
        mean_mag = float(np.mean(mag))
        # Vector coherence in [0, 1]: 1.0 means uniform camera pan, 0.0 means random/jittery vectors
        sum_vec_mag = float(np.hypot(np.mean(dx), np.mean(dy)))
        coherence = sum_vec_mag / (mean_mag + 1e-6) if mean_mag > 1e-4 else 0.0
        motion_mags.append(mean_mag)
        motion_coherence.append(float(np.clip(coherence, 0.0, 1.0)))

    # Second-order temporal finite differences isolate alternating flicker (A -> B -> A)
    # while ignoring smooth linear pans or gradual lighting fades.
    hf_osc = np.abs(hf_energies[2:] - 2.0 * hf_energies[1:-1] + hf_energies[:-2])
    luma_osc = np.abs(mean_lumas[2:] - 2.0 * mean_lumas[1:-1] + mean_lumas[:-2])
    lap_scale = float(np.median(lap_energies)) + 1e-4
    lap_osc = np.abs(lap_energies[2:] - 2.0 * lap_energies[1:-1] + lap_energies[:-2]) / lap_scale

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for k in range(len(hf_osc)):
        idx = k + 1
        # Average coherent motion across the two adjacent transitions around frame `idx`
        local_coherent_motion = 0.5 * (
            motion_mags[k] * motion_coherence[k]
            + motion_mags[k + 1] * motion_coherence[k + 1]
        )
        damping = 1.0 / (1.0 + motion_damping * local_coherent_motion)
        raw_osc = 0.50 * float(hf_osc[k]) + 0.35 * float(luma_osc[k]) + 0.15 * min(float(lap_osc[k]), 2.0)
        frame_scores[idx] = raw_osc * damping

    clip_score = float(np.percentile(frame_scores[1:-1], 90))
    return {
        "hf_energies": [round(float(x), 6) for x in hf_energies],
        "motion_magnitudes": [round(float(x), 4) for x in motion_mags],
        "motion_coherence": [round(float(x), 4) for x in motion_coherence],
        "frame_flicker_scores": [round(float(x), 6) for x in frame_scores],
        "clip_score": round(clip_score, 6),
    }


class FlickerFilter(BaseFilter):
    """Pipeline filter flagging clips with excessive motion-compensated temporal flicker."""

    name = "flicker"

    def __init__(
        self,
        threshold: float = 0.045,
        cutoff_ratio: float = 0.25,
        motion_damping: float = 0.25,
        use_cv2: bool = True,
    ) -> None:
        self.threshold = threshold
        self.cutoff_ratio = cutoff_ratio
        self.motion_damping = motion_damping
        self.use_cv2 = use_cv2

    def evaluate(self, frames: np.ndarray, fps: float = 24.0) -> FilterResult:
        analysis = compute_motion_compensated_flicker(
            frames,
            cutoff_ratio=self.cutoff_ratio,
            motion_damping=self.motion_damping,
            use_cv2=self.use_cv2,
        )
        frame_scores = analysis["frame_flicker_scores"]
        flagged = [i for i, s in enumerate(frame_scores) if s > self.threshold]
        clip_score = float(analysis["clip_score"])
        passed = clip_score <= self.threshold and len(flagged) == 0

        return FilterResult(
            filter_name=self.name,
            passed=passed,
            score=clip_score,
            threshold=self.threshold,
            flagged_frames=flagged,
            metrics={
                "fps": fps,
                "mean_hf_energy": round(float(np.mean(analysis["hf_energies"])), 6)
                if analysis["hf_energies"]
                else 0.0,
                "mean_motion_magnitude": round(float(np.mean(analysis["motion_magnitudes"])), 4)
                if analysis["motion_magnitudes"]
                else 0.0,
                "max_frame_score": round(float(max(frame_scores)), 6) if frame_scores else 0.0,
            },
        )
