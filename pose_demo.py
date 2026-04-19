"""
Phase 1 — step 1: minimal pose demo.

Open the webcam, run MediaPipe's pose model on each frame,
draw the skeleton, show it in a window. Press 'q' to quit.
"""

import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Seconds the T-key countdown waits before firing.
COUNTDOWN_SECONDS = 5.0
# Folder where snapshots get saved. Anchored to this file so it lands in the
# project directory no matter where you launch python3 from.
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


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
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

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

# None = no countdown running. Otherwise, the Unix timestamp at which the
# countdown should fire and save a frame.
countdown_until = None

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

        # Highlight the three landmarks the knee-angle math actually uses.
        # A big red circle + a letter label makes it obvious at a glance
        # whether H/K/A landed on the right body parts.
        for idx, label in [(RIGHT_HIP, "H"), (RIGHT_KNEE, "K"), (RIGHT_ANKLE, "A")]:
            lm = person[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (px, py), 9, (0, 0, 255), 2)
            cv2.putText(
                frame_bgr, label, (px + 12, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        # Pull out hip, knee, ankle as (x, y) pairs and compute the knee angle.
        hip = (person[RIGHT_HIP].x, person[RIGHT_HIP].y)
        knee = (person[RIGHT_KNEE].x, person[RIGHT_KNEE].y)
        ankle = (person[RIGHT_ANKLE].x, person[RIGHT_ANKLE].y)
        knee_angle = angle_between(hip, knee, ankle)

        # MediaPipe's own confidence that each landmark is actually visible.
        vis_hip = person[RIGHT_HIP].visibility
        vis_knee = person[RIGHT_KNEE].visibility
        vis_ankle = person[RIGHT_ANKLE].visibility
        trustworthy = min(vis_hip, vis_knee, vis_ankle) >= MIN_VISIBILITY
        ready = trustworthy and knee_angle is not None

        if ready:
            cv2.putText(
                frame_bgr,
                f"R knee: {knee_angle:5.1f} deg",
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

    cv2.imshow("Pose demo (T=timed snap, SPACE=snap now, Q=quit)", frame_bgr)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        countdown_until = time.time() + COUNTDOWN_SECONDS
    if key == ord(" "):
        try_snapshot(frame_bgr, ready)


# --- 4. Clean up --------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
landmarker.close()
