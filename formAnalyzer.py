"""
Pose analyzer — Phase 1 snapshots + Phase 2 CSV recording.

Opens the webcam, runs MediaPipe's pose model on each frame, computes the
knee angle on whichever side you're tracking, and either saves one-off
snapshots or streams per-frame features to a CSV for later ML use.

Controls:
    T      start a 5-second countdown, then save the frame to snapshots/
    SPACE  save the current frame immediately
    L      toggle between tracking the right and left leg
    R      start/stop recording a session CSV in recordings/
    1-3    select label for the next recording (good / overstride / excessive_lean)
    Q      quit
"""

import csv
import math
import os
import time
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Seconds the T-key countdown waits before firing.
COUNTDOWN_SECONDS = 5.0
# Folders where snapshots and recordings land. Anchored to this file so they
# sit in the project directory no matter where you launch python3 from.
_HERE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(_HERE, "snapshots")
RECORDING_DIR = os.path.join(_HERE, "recordings")
CSV_COLUMNS = [
    "frame_index", "elapsed_ms", "tracking_side",
    "knee_angle", "trunk_lean", "foot_offset", "cadence_spm",
    "vis_hip", "vis_knee", "vis_ankle", "vis_shoulder", "ready", "label",
]

# Cadence detection tuning.
ANKLE_HISTORY_SECONDS = 1.5   # how far back to keep ankle-y samples
CADENCE_WINDOW_SECONDS = 6.0  # recent window for smoothing cadence
MIN_CONTACT_INTERVAL = 0.25   # refractory: no two contacts within this time

# Session-label shortcuts. Press the digit key during the live demo to change
# which label the next recording will stamp onto every row. Extend this dict
# as you collect more form categories.
LABELS = {
    ord("1"): "good",
    ord("2"): "overstride",
    ord("3"): "excessive_lean",
}


def angle_between(a, b, c):
    """Return the angle (in degrees) at vertex b, formed by points a-b-c.

    Each of a, b, c is an (x, y) pair. We build two vectors that both
    start at b (one pointing to a, one pointing to c), then use the
    dot-product formula to get the angle between them.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return None
    # Clamp to [-1, 1] to guard against tiny floating-point drift.
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


# Landmark indices we care about (MediaPipe's standard numbering).
# Each leg is a (hip, knee, ankle) triple.
LEG_SIDES = {
    "right": (24, 26, 28),
    "left": (23, 25, 27),
}
# Shoulder and hip indices used to compute torso-based features.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24

# Minimum MediaPipe `visibility` score required on hip, knee, and ankle for us
# to trust the angle. Below this we show a warning instead of a number.
MIN_VISIBILITY = 0.5


# --- 1. Load the pose model ---------------------------------------------------
# The Tasks API wants us to build a "PoseLandmarker" object, which wraps the
# trained model. We tell it which .task file to load and that we'll be
# feeding it live video frames (running_mode=VIDEO).
MODEL_PATH = "models/pose_landmarker_lite.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)


# --- 2. Open the webcam -------------------------------------------------------
# cv2.VideoCapture(0) grabs the default camera. If you had a USB webcam you
# wanted to use instead, you'd pass 1, 2, etc.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")


# --- 3. Main loop: read frame -> detect pose -> draw -> show ------------------
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

# Which leg to analyze. Toggled at runtime with the L key.
tracking_side = "right"

# Label written into every row of the next recording. Change with 1-3.
current_label = "good"

# None = no countdown running. Otherwise, the Unix timestamp at which the
# countdown should fire and save a frame.
countdown_until = None

# Recording state. While a session is active, `recording_file` holds an open
# CSV file handle, `recording_writer` a csv.writer bound to it, and we track
# when it started (for elapsed_ms) and how many rows we've written so far.
recording_file = None
recording_writer = None
recording_started_at = None
recording_row_count = 0
recording_ready_count = 0
recording_contact_count = 0

# Cadence detection state. `ankle_history` is a rolling window of
# (time, ankle_y) tuples; `contact_times` holds the timestamps of detected
# foot contacts (local maxes in ankle_y) within CADENCE_WINDOW_SECONDS.
ankle_history = deque()
contact_times = deque()
last_contact_time = 0.0

# Ephemeral on-screen banner. None or (text, expires_at, color) where color
# is a BGR tuple. Rendered each frame while time.time() < expires_at.
status_message = None
STATUS_DURATION = 1.5  # seconds a status banner stays visible

def show_status(text, color):
    """Set the on-screen banner for the next STATUS_DURATION seconds."""
    global status_message
    status_message = (text, time.time() + STATUS_DURATION, color)

def save_snapshot(image):
    """Write the given frame to the snapshots folder with a unix-timestamp name."""
    path = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg")
    cv2.imwrite(path, image)
    print(f"Saved snapshot: {path}")
    show_status("Snapshot saved", (0, 255, 0))

def try_snapshot(image, ready_now):
    """Save if the pose is trustworthy right now; otherwise show an error."""
    if ready_now:
        save_snapshot(image)
    else:
        show_status("Snapshot failed: pose not ready", (0, 0, 255))

def start_recording():
    """Open a fresh CSV, write the header row, and arm recording state."""
    global recording_file, recording_writer, recording_started_at
    global recording_row_count, recording_ready_count, recording_contact_count
    path = os.path.join(RECORDING_DIR, f"session_{int(time.time())}.csv")
    recording_file = open(path, "w", newline="")
    recording_writer = csv.writer(recording_file)
    recording_writer.writerow(CSV_COLUMNS)
    recording_started_at = time.time()
    recording_row_count = 0
    recording_ready_count = 0
    recording_contact_count = 0
    print(f"Recording started: {path}")
    show_status(f"Recording: {os.path.basename(path)}", (0, 200, 255))

def stop_recording():
    """Flush and close the CSV, then tell the user how many rows landed."""
    global recording_file, recording_writer, recording_started_at
    if recording_file is None:
        return
    path = recording_file.name
    recording_file.close()
    recording_file = None
    recording_writer = None
    recording_started_at = None
    print(
        f"Recording stopped: {path} "
        f"({recording_row_count} rows, {recording_ready_count} ready, "
        f"{recording_contact_count} contacts)"
    )
    show_status(
        f"Saved {recording_row_count} rows, {recording_contact_count} contacts",
        (0, 255, 0),
    )

frame_index = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    # MediaPipe wants RGB, OpenCV gives us BGR. Convert.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Wrap the raw pixels in a MediaPipe Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Run pose detection. For VIDEO mode we pass a timestamp in milliseconds;
    # using the frame index is fine for a live demo.
    result = landmarker.detect_for_video(mp_image, frame_index)
    frame_index += 1

    # Reset each frame. Flipped to True below only if all three landmarks we
    # rely on came back with enough confidence.
    ready = False

    # If a person was found, draw their 33 keypoints on the original frame.
    if result.pose_landmarks:
        h, w = frame_bgr.shape[:2]
        person = result.pose_landmarks[0]

        for landmark in person:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)

        # Pick the three landmark indices for whichever leg we're tracking.
        hip_idx, knee_idx, ankle_idx = LEG_SIDES[tracking_side]
        side_letter = "R" if tracking_side == "right" else "L"

        # Highlight the three landmarks the knee-angle math actually uses.
        # A big red circle + a letter label makes it obvious at a glance
        # whether H/K/A landed on the right body parts.
        for idx, label in [(hip_idx, "H"), (knee_idx, "K"), (ankle_idx, "A")]:
            lm = person[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (px, py), 9, (0, 0, 255), 2)
            cv2.putText(
                frame_bgr, label, (px + 12, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        # Pull out hip, knee, ankle as (x, y) pairs and compute the knee angle.
        hip = (person[hip_idx].x, person[hip_idx].y)
        knee = (person[knee_idx].x, person[knee_idx].y)
        ankle = (person[ankle_idx].x, person[ankle_idx].y)
        knee_angle = angle_between(hip, knee, ankle)

        # MediaPipe's own confidence that each landmark is actually visible.
        vis_hip = person[hip_idx].visibility
        vis_knee = person[knee_idx].visibility
        vis_ankle = person[ankle_idx].visibility
        trustworthy = min(vis_hip, vis_knee, vis_ankle) >= MIN_VISIBILITY
        ready = trustworthy and knee_angle is not None

        if ready:
            cv2.putText(
                frame_bgr,
                f"{side_letter} knee: {knee_angle:5.1f} deg",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame_bgr,
                f"Low confidence (H={vis_hip:.2f} K={vis_knee:.2f} A={vis_ankle:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        # Torso midpoints for trunk-lean and normalization.
        l_sh, r_sh = person[LEFT_SHOULDER], person[RIGHT_SHOULDER]
        l_hp, r_hp = person[LEFT_HIP_IDX], person[RIGHT_HIP_IDX]
        shoulder_mid = ((l_sh.x + r_sh.x) / 2, (l_sh.y + r_sh.y) / 2)
        hip_mid = ((l_hp.x + r_hp.x) / 2, (l_hp.y + r_hp.y) / 2)
        vis_shoulder = min(l_sh.visibility, r_sh.visibility)

        # Trunk lean: angle (degrees) of the hip->shoulder vector from
        # vertical. 0 = upright, positive = tilted in either direction.
        # atan2(|dx|, -dy) because image y grows downward, so "up" is -dy.
        dx_t = shoulder_mid[0] - hip_mid[0]
        dy_t = shoulder_mid[1] - hip_mid[1]
        torso_len = math.hypot(dx_t, dy_t)
        trunk_lean = (
            math.degrees(math.atan2(abs(dx_t), -dy_t)) if torso_len > 1e-6 else None
        )

        # Foot offset: signed horizontal distance from hip to ankle,
        # normalized by torso length so the number doesn't change with
        # camera distance. Positive = ankle in front of hip in image
        # coords; negative = behind.
        foot_offset = (ankle[0] - hip[0]) / torso_len if torso_len > 1e-6 else None

        # Secondary on-screen line: trunk + foot offset.
        if trunk_lean is not None and foot_offset is not None:
            cv2.putText(
                frame_bgr,
                f"Trunk: {trunk_lean:4.1f} deg   Foot: {foot_offset:+.2f}",
                (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )

        # Cadence detection — look for local maxima in ankle.y over time.
        cadence_spm = None
        now = time.time()
        ankle_history.append((now, ankle[1]))
        while ankle_history and now - ankle_history[0][0] > ANKLE_HISTORY_SECONDS:
            ankle_history.popleft()

        # Need at least 7 samples so we can look 3 on either side of a
        # candidate. Candidate is the sample 3 frames back from the end.
        if len(ankle_history) >= 7:
            hist = list(ankle_history)
            mid = len(hist) - 4
            mid_t, mid_y = hist[mid]
            left_max = max(s[1] for s in hist[mid - 3:mid])
            right_max = max(s[1] for s in hist[mid + 1:mid + 4])
            # Candidate is a local max if it beats both neighborhoods
            # AND we haven't just registered one (refractory).
            if (mid_y > left_max and mid_y > right_max
                    and now - last_contact_time > MIN_CONTACT_INTERVAL):
                contact_times.append(mid_t)
                last_contact_time = now
                if recording_file is not None:
                    recording_contact_count += 1

        # Drop contacts older than our averaging window.
        while contact_times and now - contact_times[0] > CADENCE_WINDOW_SECONDS:
            contact_times.popleft()

        # Need 2+ contacts to measure an interval. Multiply by 2 because we
        # only track one leg, and true cadence counts both feet.
        if len(contact_times) >= 2:
            span = contact_times[-1] - contact_times[0]
            if span > 0:
                cadence_spm = (len(contact_times) - 1) / span * 60 * 2

        if cadence_spm is not None:
            cv2.putText(
                frame_bgr,
                f"Cadence: {cadence_spm:.0f} spm",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )

        # If a recording session is active, append this frame's features.
        if recording_writer is not None:
            if ready:
                recording_ready_count += 1
            elapsed_ms = int((time.time() - recording_started_at) * 1000)
            recording_writer.writerow([
                frame_index,
                elapsed_ms,
                tracking_side,
                f"{knee_angle:.2f}" if knee_angle is not None else "",
                f"{trunk_lean:.2f}" if trunk_lean is not None else "",
                f"{foot_offset:.3f}" if foot_offset is not None else "",
                f"{cadence_spm:.1f}" if cadence_spm is not None else "",
                f"{vis_hip:.3f}",
                f"{vis_knee:.3f}",
                f"{vis_ankle:.3f}",
                f"{vis_shoulder:.3f}",
                1 if ready else 0,
                current_label,
            ])
            recording_row_count += 1

    # Readiness border: green if the three key landmarks are confident,
    # red otherwise. Drawn as a thick rectangle hugging the frame edges.
    border_color = (0, 255, 0) if ready else (0, 0, 255)
    cv2.rectangle(
        frame_bgr,
        (0, 0),
        (frame_bgr.shape[1] - 1, frame_bgr.shape[0] - 1),
        border_color,
        thickness=12,
    )

    # REC indicator — bright red text with elapsed time, row count, and
    # quality signals so you can tell from across the room whether the
    # session is actually capturing useful frames.
    if recording_file is not None:
        elapsed_s = time.time() - recording_started_at
        rec_text = f"REC  {elapsed_s:5.1f}s  rows={recording_row_count}"
        cv2.putText(
            frame_bgr, rec_text, (20, 144),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        pct = (100 * recording_ready_count / recording_row_count) if recording_row_count else 0
        stats_text = (
            f"ready={recording_ready_count} ({pct:.0f}%)   "
            f"contacts={recording_contact_count}"
        )
        cv2.putText(
            frame_bgr, stats_text, (20, 174),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # Top-right "Tracking: X Leg" indicator. Right-align by measuring the
    # text width with getTextSize, then placing its left edge at
    # frame_width - text_width - margin.
    tracking_label = f"Tracking: {tracking_side.capitalize()} Leg"
    (tl_w, tl_h), _ = cv2.getTextSize(
        tracking_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2,
    )
    margin = 20
    tx = frame_bgr.shape[1] - tl_w - margin
    ty = margin + tl_h
    cv2.putText(
        frame_bgr, tracking_label, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA,
    )

    # Second top-right line: the label that will be stamped on rows of the
    # next recording. Right-aligned the same way.
    label_text = f"Label: {current_label}"
    (lbl_w, lbl_h), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
    )
    cv2.putText(
        frame_bgr, label_text,
        (frame_bgr.shape[1] - lbl_w - margin, ty + lbl_h + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA,
    )

    # If a countdown is running, overlay the remaining seconds; when it hits
    # zero, save this frame (overlays and all) and clear the countdown.
    if countdown_until is not None:
        remaining = countdown_until - time.time()
        if remaining > 0:
            cv2.putText(
                frame_bgr,
                f"Snapshot in {remaining:.1f}s",
                (12, frame_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            try_snapshot(frame_bgr, ready)
            countdown_until = None

    # Render the status banner, if one is active and hasn't expired.
    if status_message is not None:
        text, expires_at, color = status_message
        if time.time() < expires_at:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            fh, fw = frame_bgr.shape[:2]
            x = (fw - tw) // 2
            y = fh // 2
            # Translucent dark backdrop so the text is readable on any scene.
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x - 16, y - th - 12), (x + tw + 16, y + 12), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)
            cv2.putText(frame_bgr, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        else:
            status_message = None

    cv2.imshow(
        "Pose analyzer (T=snap, SPACE=snap now, L=swap leg, R=record, 1-3=label, Q=quit)",
        frame_bgr,
    )
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        countdown_until = time.time() + COUNTDOWN_SECONDS
    if key == ord(" "):
        try_snapshot(frame_bgr, ready)
    if key == ord("l"):
        tracking_side = "left" if tracking_side == "right" else "right"
        # Reset gait state — the other leg's ankle baseline is different, so
        # peak detection would be confused for a moment otherwise.
        ankle_history.clear()
        contact_times.clear()
        last_contact_time = 0.0
        show_status(f"Now tracking: {tracking_side.capitalize()} Leg", (255, 255, 0))
    if key == ord("r"):
        if recording_file is None:
            start_recording()
        else:
            stop_recording()
    if key in LABELS:
        current_label = LABELS[key]
        show_status(f"Label: {current_label}", (200, 255, 200))


# --- 4. Clean up --------------------------------------------------------------
# Close an in-progress CSV cleanly before tearing down, otherwise you'd lose
# buffered rows on exit.
stop_recording()
cap.release()
cv2.destroyAllWindows()
landmarker.close()
