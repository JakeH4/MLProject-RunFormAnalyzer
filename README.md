# Running Form Analyzer

A computer-vision tool that tracks a runner's body keypoints from side-view video and measures form-relevant joint angles. Built as a coursework project for CSCI 4622 (Machine Learning) at CU Boulder.

## Motivation

As a marathon runner with a history of recurring injuries, I've learned firsthand that subtle differences in running mechanics — overstriding, excessive heel strike, poor trunk lean — are major contributors to injury risk. Most runners have no easy way to get objective feedback on their mechanics. This project is the foundation for a tool that lets any runner point a camera at themselves and get actionable form feedback.

## Current status (Phases 1–5 complete, data-collection iteration ongoing)

- Live webcam pose estimation via MediaPipe (33 body keypoints).
- Four form-relevant features computed per frame: knee angle, trunk lean, foot offset, cadence (spm).
- Per-landmark visibility confidence is checked before displaying metrics. A green/red border around the camera view tells you at a glance whether the current frame is trustworthy.
- One-off snapshot capture plus bulk CSV recording (optionally with an overlay-baked MP4) for dataset collection.
- Per-session form labels stamped into every CSV row — digit keys 1–3 select `good`, `overstride`, or `excessive_lean`.
- Baseline random-forest classifier in `train.py`, reporting per-class accuracy, confusion matrix, and feature importances.

## Demo

[`demos/demo_good_1776620494.mp4`](demos/demo_good_1776620494.mp4) shows the analyzer running live in an outdoor setting, with pose-skeleton overlays and all four measured features visible on screen as I jog past the camera.

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
python3 formAnalyzer.py
```

Controls:

- **T** — start a 5-second countdown, then save a frame to `snapshots/`.
- **SPACE** — save a frame immediately.
- **L** — toggle tracking between the right and left leg.
- **R** — start/stop a bulk recording session (CSV of per-frame features) in `recordings/`.
- **V** — toggle video capture on; the next recording also writes an overlaid `.mp4` alongside the CSV.
- **1–3** — pick the label stamped on rows of the next recording: `good`, `overstride`, `excessive_lean`.
- **Q** — quit.

A snapshot or a ready-marked CSV row is only produced when the tracked hip, knee, and ankle are all above the visibility threshold (green border). Otherwise you get an on-screen "pose not ready" warning.

## Training

Once you have labeled recordings in `recordings/`, run:

```bash
python3 train.py
```

The script concatenates every labeled session, filters to trustworthy frames, does a stratified 80/20 split, fits a 100-tree random forest, and prints per-class precision/recall/F1, a confusion matrix, and feature importances.

## Roadmap

- **Phase 1** ✅ Webcam pose estimation with knee-angle measurement.
- **Phase 2** ✅ Bulk per-frame CSV recording.
- **Phase 3** ✅ Extra features: trunk lean, foot offset, cadence.
- **Phase 4** ✅ Per-session form labels stamped into each row.
- **Phase 5** ✅ Baseline random-forest classifier with accuracy + feature-importance reporting.
- **Ongoing** — Multi-session data collection for honest session-level cross-validation, plus derived features like `foot_offset_at_contact` for better overstride separation.
- **Future** — Outdoor running validation on real (not deliberately-exaggerated) form; comparison against a small neural network.

## Tech

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — pretrained pose landmark detection.
- OpenCV — video I/O and overlay drawing.
- Python 3.13.
