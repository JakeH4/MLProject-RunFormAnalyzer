# Running Form Analyzer

A computer-vision tool that tracks a runner's body keypoints from side-view video and measures form-relevant joint angles. Built as a coursework project for CSCI 4622 (Machine Learning) at CU Boulder.

## Motivation

As a marathon runner with a history of recurring injuries, I've learned firsthand that subtle differences in running mechanics — overstriding, excessive heel strike, poor trunk lean — are major contributors to injury risk. Most runners have no easy way to get objective feedback on their mechanics. This project is the foundation for a tool that lets any runner point a camera at themselves and get actionable form feedback.

## Current status (Phase 1)

- Live webcam pose estimation via MediaPipe (33 body keypoints).
- Right-leg knee angle computed per frame from hip–knee–ankle coordinates.
- Per-landmark visibility confidence is checked before displaying an angle. A green/red border around the camera view tells you at a glance whether the current frame is trustworthy.
- Timed and on-demand snapshot capture, saved to `snapshots/` with overlays baked in.

## Setup

Requires Python 3.10+.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download the pose model (~5 MB)
mkdir -p models
curl -L -o models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

## Usage

```bash
python3 poseAnalyzer.py
```

Controls:

- **T** — start a 5-second countdown, then save a frame to `snapshots/`.
- **SPACE** — save a frame immediately.
- **Q** — quit.

A snapshot is only saved when the right hip, knee, and ankle are all tracked above the visibility threshold (green border). Otherwise you get an on-screen "pose not ready" warning and no file is written.

## Roadmap

- **Phase 2** — Bulk recording mode: export a clip's per-frame features to CSV.
- **Phase 3** — Additional features: trunk lean, foot strike angle, cadence.
- **Phase 4** — Hand-labeled dataset of form categories (good / overstriding / excessive heel strike / etc.).
- **Phase 5** — Train a classifier (random forest, then a small neural network) on the engineered pose features.

## Tech

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — pretrained pose landmark detection.
- OpenCV — video I/O and overlay drawing.
- Python 3.13.
