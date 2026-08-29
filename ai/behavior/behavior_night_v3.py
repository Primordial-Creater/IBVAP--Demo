import cv2
import json
import os
import time
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V3
# Robust Night Detection + Behavioral Analytics
# ============================================================

INPUT_VIDEO = "data/test/test _face_detector4.mp4"

MODEL_PATH = "yolo11m.pt"

OUTPUT_DIR = f"runs/ibvap/behavior_night_v3/behavior_v3_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "ibvap_behavior_night_v3.mp4"
)

EVENT_LOG = os.path.join(
    OUTPUT_DIR,
    "ibvap_behavior_events_v3.jsonl"
)


# ============================================================
# PARAMETERS
# ============================================================

CONFIDENCE = 0.35

# Night detection
NIGHT_ENTER_BRIGHTNESS = 72
NIGHT_EXIT_BRIGHTNESS = 100

NIGHT_ENTER_DARK_RATIO = 0.55
NIGHT_EXIT_DARK_RATIO = 0.40

NIGHT_ENTER_FRAMES = 15
NIGHT_EXIT_FRAMES = 30

EMA_ALPHA = 0.08

# Movement
MOVEMENT_WINDOW = 12
NIGHT_MOVEMENT_DISTANCE = 18

# Rapid movement
RAPID_MOVEMENT_DISTANCE = 45

# Loitering
LOITER_DISTANCE = 35
LOITER_FRAMES = 120

# Prevent repeated events
EVENT_COOLDOWN = 100


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V3")
print("=" * 70)

print("[INFO] Loading YOLO M model...")

model = YOLO(MODEL_PATH)

print("[INFO] YOLO M model loaded.")

device = 0

if hasattr(model, "predict"):
    print("[INFO] CUDA GPU enabled.")


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Unable to open video: {INPUT_VIDEO}")


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACK STATE
# ============================================================

positions = defaultdict(lambda: deque(maxlen=MOVEMENT_WINDOW))

loiter_counter = defaultdict(int)

last_event_frame = defaultdict(lambda: -999999)


# ============================================================
# NIGHT STATE
# ============================================================

night_state = False

night_enter_counter = 0
night_exit_counter = 0

brightness_ema = None

night_frames = 0


# ============================================================
# EVENT COUNTERS
# ============================================================

night_movement_events = 0
loitering_events = 0
rapid_movement_events = 0
suspicious_events = 0


# ============================================================
# EVENT LOGGER
# ============================================================

def log_event(event_type, track_id, frame_number, timestamp, extra=None):

    event = {
        "event_type": event_type,
        "track_id": int(track_id),
        "frame": int(frame_number),
        "timestamp_seconds": round(timestamp, 3)
    }

    if extra:
        event.update(extra)

    with open(EVENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ============================================================
# NIGHT DETECTION
# ============================================================

def calculate_night_metrics(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Remove extreme highlights.
    # This prevents a small bright line/light from dominating
    # the measurement.
    valid_pixels = gray[(gray > 5) & (gray < 245)]

    if len(valid_pixels) == 0:
        return 0, 1

    median_brightness = float(
        __import__("numpy").median(valid_pixels)
    )

    dark_ratio = float(
        (__import__("numpy").sum(valid_pixels < 80))
        / len(valid_pixels)
    )

    return median_brightness, dark_ratio


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    timestamp = frame_number / fps


    # --------------------------------------------------------
    # NIGHT METRICS
    # --------------------------------------------------------

    brightness, dark_ratio = calculate_night_metrics(frame)

    if brightness_ema is None:
        brightness_ema = brightness
    else:
        brightness_ema = (
            EMA_ALPHA * brightness
            + (1 - EMA_ALPHA) * brightness_ema
        )


    night_candidate = (
        brightness_ema < NIGHT_ENTER_BRIGHTNESS
        and dark_ratio > NIGHT_ENTER_DARK_RATIO
    )

    day_candidate = (
        brightness_ema > NIGHT_EXIT_BRIGHTNESS
        or dark_ratio < NIGHT_EXIT_DARK_RATIO
    )


    # --------------------------------------------------------
    # NIGHT STATE MACHINE
    # --------------------------------------------------------

    if not night_state:

        if night_candidate:

            night_enter_counter += 1

        else:

            night_enter_counter = 0


        if night_enter_counter >= NIGHT_ENTER_FRAMES:

            night_state = True

            night_enter_counter = 0

            print(
                f"[INFO] NIGHT STATE ENTERED | Frame {frame_number}"
            )


    else:

        if day_candidate:

            night_exit_counter += 1

        else:

            night_exit_counter = 0


        if night_exit_counter >= NIGHT_EXIT_FRAMES:

            night_state = False

            night_exit_counter = 0

            print(
                f"[INFO] DAY STATE ENTERED | Frame {frame_number}"
            )


    if night_state:

        night_frames += 1


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        classes=[0],
        device=device,
        verbose=False
    )


    suspicious_this_frame = set()


    # ========================================================
    # PERSON ANALYSIS
    # ========================================================

    if results and results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            ids = boxes.id.int().cpu().tolist()

            xyxy = boxes.xyxy.cpu().tolist()

            confidences = boxes.conf.cpu().tolist()


            for track_id, box, confidence in zip(
                ids,
                xyxy,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                cx = int((x1 + x2) / 2)

                cy = int((y1 + y2) / 2)


                # ------------------------------------------------
                # POSITION HISTORY
                # ------------------------------------------------

                positions[track_id].append((cx, cy))


                movement = 0

                if len(positions[track_id]) >= 2:

                    old_x, old_y = positions[track_id][0]

                    movement = (
                        ((cx - old_x) ** 2)
                        + ((cy - old_y) ** 2)
                    ) ** 0.5


                # ------------------------------------------------
                # LOITERING
                # ------------------------------------------------

                if movement < LOITER_DISTANCE:

                    loiter_counter[track_id] += 1

                else:

                    loiter_counter[track_id] = 0


                if (
                    loiter_counter[track_id] >= LOITER_FRAMES
                    and frame_number - last_event_frame[track_id]
                    > EVENT_COOLDOWN
                ):

                    log_event(
                        "LOITERING",
                        track_id,
                        frame_number,
                        timestamp,
                        {
                            "confidence": round(
                                float(confidence), 4
                            )
                        }
                    )

                    print(
                        f"[ALERT] LOITERING | "
                        f"Track ID: {track_id} | "
                        f"Frame: {frame_number}"
                    )

                    loitering_events += 1

                    suspicious_this_frame.add(track_id)

                    last_event_frame[track_id] = frame_number


                # ------------------------------------------------
                # RAPID MOVEMENT
                # ------------------------------------------------

                if (
                    movement >= RAPID_MOVEMENT_DISTANCE
                    and frame_number - last_event_frame[track_id]
                    > EVENT_COOLDOWN
                ):

                    log_event(
                        "RAPID_MOVEMENT",
                        track_id,
                        frame_number,
                        timestamp,
                        {
                            "movement_pixels": round(
                                movement, 2
                            ),
                            "confidence": round(
                                float(confidence), 4
                            )
                        }
                    )

                    print(
                        f"[ALERT] RAPID_MOVEMENT | "
                        f"Track ID: {track_id} | "
                        f"Frame: {frame_number}"
                    )

                    rapid_movement_events += 1

                    suspicious_this_frame.add(track_id)

                    last_event_frame[track_id] = frame_number


                # ------------------------------------------------
                # NIGHT MOVEMENT
                # ------------------------------------------------

                if (
                    night_state
                    and movement >= NIGHT_MOVEMENT_DISTANCE
                    and frame_number - last_event_frame[track_id]
                    > EVENT_COOLDOWN
                ):

                    log_event(
                        "NIGHT_MOVEMENT",
                        track_id,
                        frame_number,
                        timestamp,
                        {
                            "movement_pixels": round(
                                movement, 2
                            ),
                            "brightness": round(
                                brightness_ema, 2
                            )
                        }
                    )

                    print(
                        f"[ALERT] NIGHT_MOVEMENT | "
                        f"Track ID: {track_id} | "
                        f"Frame: {frame_number}"
                    )

                    night_movement_events += 1

                    suspicious_this_frame.add(track_id)

                    last_event_frame[track_id] = frame_number


                # ------------------------------------------------
                # DRAW PERSON
                # ------------------------------------------------

                if track_id in suspicious_this_frame:

                    label = f"SUSPICIOUS ID:{track_id}"

                elif night_state:

                    label = f"NIGHT ID:{track_id}"

                else:

                    label = f"ID:{track_id}"


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )


    # ========================================================
    # COMPOSITE SUSPICIOUS EVENT
    # ========================================================

    for track_id in suspicious_this_frame:

        if frame_number - last_event_frame[track_id] > 2:

            log_event(
                "SUSPICIOUS_ACTIVITY",
                track_id,
                frame_number,
                timestamp
            )

            suspicious_events += 1


    # ========================================================
    # STATUS PANEL
    # ========================================================

    state_text = "NIGHT" if night_state else "DAY"

    cv2.rectangle(
        frame,
        (10, 10),
        (360, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"ENVIRONMENT: {state_text}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Brightness: {brightness_ema:.1f}",
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Night Movement: {night_movement_events}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    writer.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V3 COMPLETE")
print("=" * 70)

print(f"Resolution        : {width}x{height}")
print(f"Frames processed  : {frame_number}")
print(f"Night frames      : {night_frames}")
print(f"Night events      : {night_movement_events}")
print(f"Loitering events  : {loitering_events}")
print(f"Rapid movement    : {rapid_movement_events}")
print(f"Suspicious events : {suspicious_events}")

print()
print(f"Output video      : {OUTPUT_VIDEO}")
print(f"Event log         : {EVENT_LOG}")

print("=" * 70)