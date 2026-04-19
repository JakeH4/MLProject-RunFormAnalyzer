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

## Results

Ten labeled sessions were collected — indoor and outdoor across multiple camera setups — yielding 2,629 ready frames: 933 `good`, 906 `overstride`, 790 `excessive_lean`. Two evaluation protocols were run and compared:

### Naive row-level split (inflated baseline)

A stratified 80/20 random split over all rows produced **99.7% test accuracy**. That number is misleading — with only a handful of sessions per label, random row splits leak session identity into the test set, so the model learns "which session is this" rather than "what form is this."

### Leave-one-session-out cross-validation (honest)

Holding out each of the 10 sessions in turn and training on the remaining 9 yielded a much more realistic picture:

| Metric | Value |
|---|---|
| Mean accuracy across folds | **0.435** |
| Best fold | 0.920 |
| Worst fold | 0.027 |
| `excessive_lean` F1 | **0.86** |
| `good` F1 | 0.36 |
| `overstride` F1 | **0.11** |

The aggregate confusion matrix shows the specific failure mode:

```
                true\pred   excessive_lean    good    overstride
excessive_lean              682                29           79
good                         39               378          516
overstride                   74               748           84
```

`excessive_lean` is cleanly separable — trunk lean is a robust feature that generalizes across sessions. But `good` and `overstride` are effectively indistinguishable: the model misclassifies **748 of 906 overstride rows as `good`**, and **516 of 933 good rows as `overstride`**. Near-random behavior between those two classes.

### Root cause

The `foot_offset` feature is averaged across the whole gait cycle. Because the foot swings forward AND backward every stride, its per-session mean sits near zero regardless of overstriding — the signal is drowned out by its own symmetry. The real overstride signal lives at foot-contact moments specifically, not in frame-averaged foot position.

### Feature importance (model on all data)

- `trunk_lean` 0.41 — dominant, matches the strong `excessive_lean` performance.
- `foot_offset` 0.24.
- `cadence_spm` 0.19.
- `knee_angle` 0.17.

### Takeaways

1. Session-level cross-validation is non-negotiable when you have few sessions. A row-level 80/20 split can overstate true accuracy by 50+ percentage points.
2. Aggregating features across an entire signal cycle can destroy the very phase-specific information that distinguishes the classes you care about.
3. Which class a model can learn says something about the physical signal, not the model — `excessive_lean` works because trunk posture is relatively stationary; `overstride` doesn't because the relevant information is transient and lost in the mean.

## Roadmap

- **Phase 1** ✅ Webcam pose estimation with knee-angle measurement.
- **Phase 2** ✅ Bulk per-frame CSV recording.
- **Phase 3** ✅ Extra features: trunk lean, foot offset, cadence.
- **Phase 4** ✅ Per-session form labels stamped into each row.
- **Phase 5** ✅ Baseline random-forest classifier, naive-split vs. LOSO evaluation contrast.
- **Phase 6** 🚧 Gait-event feature engineering: `foot_offset_at_contact` and `knee_angle_at_contact` derived from ankle-y peak detection, expected to resolve the good/overstride confusion.
- **Future** — Outdoor running validation on real (not deliberately-exaggerated) form; comparison against a small neural network.

## Tech

- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — pretrained pose landmark detection.
- OpenCV — video I/O and overlay drawing.
- Python 3.13.
