# video-artifact-guard

Automated computer-vision quality-control filtering for AI-generated video artifacts — targeting temporal flicker, dark-frame drops, and hand/limb structural anomalies.

## Why

Generative video pipelines frequently produce clips that look plausible on a single keyframe but fail in motion: high-frequency texture strobing, sudden underexposed/crushed frames, or unstable hand silhouettes. `video-artifact-guard` provides lightweight, deterministic CV filters that gate generated clips before downstream editing or delivery.

## Architecture

The codebase is organized into three layers:

```
video-artifact-guard/
├── filters/        # Pluggable defect detectors (flicker, dark_frame, hand_anomaly)
├── pipeline/       # Video frame sampling, filter orchestration, and structured QC reports
├── harness/        # Synthetic defect generators and regression evaluation suite
└── tests/          # Unit and integration tests
```

- **`filters/`**: Self-contained signal-processing and CV filters implementing a shared `BaseFilter` interface.
  - `filters/flicker.py`: Combines 2D FFT radial high-frequency energy oscillation with motion-vector magnitude/coherence to separate genuine camera/subject motion from temporal flicker artifacts.
  - `filters/dark_frame.py`: Detects crushed-black frames and abrupt luminance drops against the clip's rolling luma baseline.
  - `filters/hand_anomaly.py`: Lightweight morphological/contour instability filter with a pluggable keypoint-detector hook (MediaPipe/RTMPose integration planned).
- **`pipeline/`**: Composable `ArtifactPipeline` that decodes frames (with configurable stride/max-frame caps), runs active filters, and outputs a JSON-serializable `ClipReport`.
- **`harness/`**: Reproducible benchmark harness (`python -m harness.eval`) that synthesizes clean panning clips and artifact-injected clips to track precision, recall, and per-clip latency as thresholds evolve.

## Quickstart

```bash
pip install -r requirements.txt
```

Run the synthetic evaluation harness:

```bash
python3 -m harness.eval
```

Run the test suite:

```bash
pytest -q
```

### Programmatic Usage

```python
import numpy as np
from filters.flicker import FlickerFilter
from filters.dark_frame import DarkFrameFilter
from pipeline.runner import ArtifactPipeline

pipe = ArtifactPipeline(filters=[FlickerFilter(), DarkFrameFilter()])
report = pipe.run_video("path/to/generated_clip.mp4")
print(report.to_dict())
```

## Status & Roadmap

- [x] High-frequency energy + motion-vector compensated flicker filter (`filters/flicker.py`)
- [x] Dark-frame & sudden luma-drop detector (`filters/dark_frame.py`)
- [x] Composable pipeline runner & synthetic regression harness (`pipeline/`, `harness/`)
- [ ] Optional ONNX / MediaPipe hand-landmark backend for `filters/hand_anomaly.py`
- [ ] Batch CLI for directory-level triage and JSONL manifest export

## License

MIT
