"""Synthetic evaluation and regression harness for video-artifact-guard."""

from harness.synthetic import (
    make_clean_pan_clip,
    make_dark_drop_clip,
    make_flicker_clip,
)

__all__ = [
    "make_clean_pan_clip",
    "make_flicker_clip",
    "make_dark_drop_clip",
]
