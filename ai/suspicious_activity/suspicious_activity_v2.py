import cv2
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


# ============================================================
# IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V2
# ============================================================

MODEL_PATH = "yolo11m.pt"

PERSON_CLASS = 0

CONFIDENCE = 0.35
IOU = 0.50

# Suspicious activity parameters
LOITERING_SECONDS = 8.0
MIN_MOVEMENT_PIXELS = 8

# Night detection
NIGHT_BRIGHTNESS_THRESHOLD = 75

# Suspicion scoring
NIGHT_PRESENCE_SCORE = 20
LOITERING_SCORE = 30
RESTRICTED_ZONE_SCORE = 50
UNUSUAL_MOVEMENT_SCORE = 20

ALERT_THRESHOLD = 50

# Tracking
TRACKER = "bytetrack.yaml"


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="IBVAP Suspicious Activity + Night Movement V2"
)

parser.add_argument(
    "--source",
    required=True,
    help="Input video path or stream URL"
)

parser.add_argument(
    "--zone",
    nargs="+",
    type=int,
    default=None,
    help="Restricted polygon points: x1 y1 x2 y2 x3 y3 ..."
)

args = parser.parse_args()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"runs/ibvap/suspicious_night_v2/{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = OUTPUT_DIR / "ibvap_suspicious_night_v2.mp4"
EVENT_LOG = OUTPUT_DIR / "ibvap_suspicious_night_events_v2.jsonl"


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V2")
print("=" * 70)

print("[INFO] Loading YOLO11m...")

model = YOLO(MODEL_PATH)

print("[INFO] YOLO11m loaded.")

# Automatically use CUDA when available
DEVICE = 0

print("[INFO] Using GPU device:", DEVICE)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(args.source)

if not cap.isOpened():
    raise RuntimeError(
        f"Unable to open video source: {args.source}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"[INFO] Resolution: {width}x{height}")
print(f"[INFO] FPS: {fps:.2f}")
print(f"[INFO] Total frames: {total_frames}")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# RESTRICTED ZONE
# ============================================================

restricted_polygon = None

if args.zone and len(args.zone) >= 6 and len(args.zone) % 2 == 0:

    points = []

    for i in range(0, len(args.zone), 2):
        points.append(
            (args.zone[i], args.zone[i + 1])
        )

    restricted_polygon = points

    print("[INFO] Restricted zone enabled:")
    print(restricted_polygon)


# ============================================================
# TRACK HISTORY
# ============================================================

track_history = {}

loitering_alerted = set()
night_alerted = set()
zone_alerted = set()

event_count = 0

start_time = time.time()


# ============================================================
# HELPERS
# ============================================================

def is_night(frame):
    """
    Estimate whether the current frame is low-light.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(gray.mean())

    return (
        brightness < NIGHT_BRIGHTNESS_THRESHOLD,
        brightness
    )


def enhance_night(frame):
    """
    Improve low-light visibility using CLAHE.
    """

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


def point_inside_polygon(point, polygon):
    """
    Check whether a point lies inside the restricted polygon.
    """

    polygon_array = __import__("numpy").array(
        polygon,
        dtype=__import__("numpy").int32
    )

    return (
        cv2.pointPolygonTest(
            polygon_array,
            (
                float(point[0]),
                float(point[1])
            ),
            False
        ) >= 0
    )


def write_event(event):
    with open(
        EVENT_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                event,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    timestamp_seconds = frame_number / fps

    # --------------------------------------------------------
    # NIGHT DETECTION
    # --------------------------------------------------------

    night_mode, brightness = is_night(frame)

    processing_frame = frame

    if night_mode:
        processing_frame = enhance_night(frame)

    # --------------------------------------------------------
    # YOLO + TRACKING
    # --------------------------------------------------------

    results = model.track(
        processing_frame,
        persist=True,
        tracker=TRACKER,
        classes=[PERSON_CLASS],
        conf=CONFIDENCE,
        iou=IOU,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DRAW RESTRICTED ZONE
    # --------------------------------------------------------

    if restricted_polygon:

        import numpy as np

        polygon_array = np.array(
            restricted_polygon,
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [polygon_array],
            True,
            (0, 0, 255),
            3
        )

    # --------------------------------------------------------
    # PROCESS TRACKS
    # --------------------------------------------------------

    current_ids = set()

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, track_id, confidence in zip(
            boxes,
            ids,
            confidences
        ):

            current_ids.add(track_id)

            x1, y1, x2, y2 = map(
                int,
                box
            )

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ------------------------------------------------
            # TRACK HISTORY
            # ------------------------------------------------

            if track_id not in track_history:

                track_history[track_id] = {
                    "first_seen": timestamp_seconds,
                    "last_seen": timestamp_seconds,
                    "positions": [],
                    "night_alerted": False,
                    "zone_alerted": False,
                    "loitering_alerted": False
                }

            history = track_history[track_id]

            history["last_seen"] = timestamp_seconds

            history["positions"].append(
                (
                    cx,
                    cy,
                    timestamp_seconds
                )
            )

            # Keep only recent history
            if len(history["positions"]) > 100:
                history["positions"].pop(0)

            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            movement = 0.0

            if len(history["positions"]) >= 2:

                old_x, old_y, _ = history["positions"][0]

                movement = (
                    (
                        (cx - old_x) ** 2
                        +
                        (cy - old_y) ** 2
                    ) ** 0.5
                )

            # ------------------------------------------------
            # LOITERING
            # ------------------------------------------------

            visible_duration = (
                timestamp_seconds
                -
                history["first_seen"]
            )

            suspicious_score = 0

            suspicious_reasons = []

            # Night-time presence
            if night_mode:

                suspicious_score += NIGHT_PRESENCE_SCORE

                suspicious_reasons.append(
                    "night_time_presence"
                )

            # Loitering
            if (
                visible_duration >= LOITERING_SECONDS
                and movement < MIN_MOVEMENT_PIXELS
            ):

                suspicious_score += LOITERING_SCORE

                suspicious_reasons.append(
                    "possible_loitering"
                )

                if not history["loitering_alerted"]:

                    history["loitering_alerted"] = True

                    event = {
                        "event_type": "SUSPICIOUS_LOITERING",
                        "track_id": int(track_id),
                        "frame": frame_number,
                        "timestamp_seconds": round(
                            timestamp_seconds,
                            3
                        ),
                        "confidence": round(
                            float(confidence),
                            4
                        ),
                        "score": suspicious_score,
                        "night_mode": night_mode
                    }

                    write_event(event)

                    event_count += 1

                    print(
                        f"[ALERT] LOITERING | "
                        f"Track ID: {track_id} | "
                        f"Frame: {frame_number}"
                    )

            # ------------------------------------------------
            # RESTRICTED ZONE
            # ------------------------------------------------

            if restricted_polygon:

                inside_zone = point_inside_polygon(
                    (cx, cy),
                    restricted_polygon
                )

                if inside_zone:

                    suspicious_score += (
                        RESTRICTED_ZONE_SCORE
                    )

                    suspicious_reasons.append(
                        "restricted_zone_intrusion"
                    )

                    if not history["zone_alerted"]:

                        history["zone_alerted"] = True

                        event = {
                            "event_type":
                                "SUSPICIOUS_ZONE_INTRUSION",
                            "track_id":
                                int(track_id),
                            "frame":
                                frame_number,
                            "timestamp_seconds":
                                round(
                                    timestamp_seconds,
                                    3
                                ),
                            "confidence":
                                round(
                                    float(confidence),
                                    4
                                ),
                            "score":
                                suspicious_score,
                            "night_mode":
                                night_mode
                        }

                        write_event(event)

                        event_count += 1

                        print(
                            f"[ALERT] ZONE INTRUSION | "
                            f"Track ID: {track_id} | "
                            f"Frame: {frame_number}"
                        )

            # ------------------------------------------------
            # NIGHT ALERT
            # ------------------------------------------------

            if (
                night_mode
                and not history["night_alerted"]
            ):

                history["night_alerted"] = True

                event = {
                    "event_type":
                        "NIGHT_TIME_MOVEMENT",
                    "track_id":
                        int(track_id),
                    "frame":
                        frame_number,
                    "timestamp_seconds":
                        round(
                            timestamp_seconds,
                            3
                        ),
                    "confidence":
                        round(
                            float(confidence),
                            4
                        ),
                    "brightness":
                        round(
                            brightness,
                            2
                        ),
                    "score":
                        NIGHT_PRESENCE_SCORE
                }

                write_event(event)

                event_count += 1

                print(
                    f"[ALERT] NIGHT MOVEMENT | "
                    f"Track ID: {track_id} | "
                    f"Frame: {frame_number}"
                )

            # ------------------------------------------------
            # FINAL SUSPICION
            # ------------------------------------------------

            if suspicious_score >= ALERT_THRESHOLD:

                event = {
                    "event_type":
                        "CONFIRMED_SUSPICIOUS_ACTIVITY",
                    "track_id":
                        int(track_id),
                    "frame":
                        frame_number,
                    "timestamp_seconds":
                        round(
                            timestamp_seconds,
                            3
                        ),
                    "confidence":
                        round(
                            float(confidence),
                            4
                        ),
                    "suspicion_score":
                        suspicious_score,
                    "reasons":
                        suspicious_reasons,
                    "night_mode":
                        night_mode
                }

                write_event(event)

            # ------------------------------------------------
            # DRAW PERSON
            # ------------------------------------------------

            if suspicious_score >= ALERT_THRESHOLD:

                box_color = (0, 0, 255)

            elif night_mode:

                box_color = (255, 0, 255)

            else:

                box_color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            label = (
                f"Person ID:{track_id} "
                f"{float(confidence):.2f}"
            )

            if night_mode:
                label += " NIGHT"

            if suspicious_score > 0:
                label += f" SCORE:{suspicious_score}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            # Center point
            cv2.circle(
                frame,
                (cx, cy),
                4,
                box_color,
                -1
            )

    # --------------------------------------------------------
    # STATUS PANEL
    # --------------------------------------------------------

    mode_text = (
        "NIGHT MODE"
        if night_mode
        else "DAY MODE"
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (330, 85),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IBVAP | {mode_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Brightness: {brightness:.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    writer.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()

elapsed = time.time() - start_time

print()
print("=" * 70)
print("IBVAP SUSPICIOUS ACTIVITY + NIGHT MOVEMENT V2 COMPLETE")
print("=" * 70)
print(f"Resolution       : {width}x{height}")
print(f"Frames processed : {frame_number}")
print(f"Events generated : {event_count}")
print(f"Processing time  : {elapsed:.2f} seconds")
print(f"Output video     : {OUTPUT_VIDEO}")
print(f"Event log        : {EVENT_LOG}")
print("=" * 70)