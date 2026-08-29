import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# IBVAP VIRTUAL FENCE V3
# ============================================================

VIDEO_PATH = "data/test/test_video.mp4"
MODEL_PATH = "yolo11n.pt"

CAMERA_ID = "CAM_001"

OUTPUT_DIR = Path("runs/ibvap/fence_v3")
OUTPUT_VIDEO = OUTPUT_DIR / "ibvap_fence_v3.mp4"
EVENT_LOG = OUTPUT_DIR / "ibvap_events_v3.jsonl"

CONFIDENCE = 0.35
DEVICE = 0

# Person must remain inside for this many frames
# before an intrusion is confirmed.
CONFIRMATION_FRAMES = 5


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# CAMERA-SPECIFIC FENCE
#
# Coordinates are normalized from 0.0 to 1.0.
# Therefore they automatically adapt to video resolution.
# ============================================================

FENCE_NORMALIZED = [
    (0.05, 0.55),
    (0.95, 0.55),
    (0.95, 0.95),
    (0.05, 0.95)
]


FENCE = [
    (
        int(x * width),
        int(y * height)
    )
    for x, y in FENCE_NORMALIZED
]

FENCE_ARRAY = np.array(
    FENCE,
    dtype=np.int32
)


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
# TRACK STATE
# ============================================================

track_state = {}

event_counter = 0
frame_number = 0


# ============================================================
# EVENT CREATOR
# ============================================================

def create_event(
    event_type,
    track_id,
    confidence,
    frame_number,
    position,
    bbox
):

    global event_counter

    event_counter += 1

    return {
        "event_id": f"EVT_{event_counter:06d}",
        "event_type": event_type,
        "severity": "HIGH",
        "camera_id": CAMERA_ID,
        "track_id": int(track_id),
        "confidence": round(float(confidence), 4),
        "frame_number": int(frame_number),
        "timestamp": datetime.now().isoformat(),
        "position": {
            "x": int(position[0]),
            "y": int(position[1])
        },
        "bbox": [
            int(v)
            for v in bbox
        ]
    }


# ============================================================
# PROCESS VIDEO
# ============================================================

with open(
    EVENT_LOG,
    "w",
    encoding="utf-8"
) as event_file:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1


        # ====================================================
        # PERSON DETECTION + TRACKING
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]


        # ====================================================
        # DRAW FENCE
        # ====================================================

        cv2.polylines(
            frame,
            [FENCE_ARRAY],
            True,
            (0, 0, 255),
            3
        )


        # ====================================================
        # PROCESS TRACKS
        # ====================================================

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            for box, track_id, confidence in zip(
                boxes,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # =================================================
                # GROUND CONTACT POINT
                # =================================================

                cx = int((x1 + x2) / 2)
                cy = int(y2)

                point = (
                    float(cx),
                    float(cy)
                )


                # =================================================
                # FENCE TEST
                # =================================================

                inside = (
                    cv2.pointPolygonTest(
                        FENCE_ARRAY,
                        point,
                        False
                    ) >= 0
                )


                # =================================================
                # INITIALIZE TRACK
                # =================================================

                if track_id not in track_state:

                    track_state[track_id] = {
                        "inside": inside,
                        "inside_frames": 0,
                        "alerted": False
                    }


                state = track_state[track_id]


                # =================================================
                # TEMPORAL CONFIRMATION
                # =================================================

                if inside:

                    state["inside_frames"] += 1

                else:

                    state["inside_frames"] = 0
                    state["alerted"] = False


                # =================================================
                # CONFIRMED INTRUSION
                # =================================================

                if (
                    inside
                    and state["inside_frames"]
                    >= CONFIRMATION_FRAMES
                    and not state["alerted"]
                ):

                    event = create_event(
                        "VIRTUAL_FENCE_INTRUSION",
                        track_id,
                        confidence,
                        frame_number,
                        (cx, cy),
                        (x1, y1, x2, y2)
                    )

                    event_file.write(
                        json.dumps(event) + "\n"
                    )

                    event_file.flush()

                    state["alerted"] = True

                    print(
                        f"[ALERT] CONFIRMED INTRUSION | "
                        f"Track ID: {track_id} | "
                        f"Frame: {frame_number}"
                    )


                state["inside"] = inside


                # =================================================
                # DRAW PERSON
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )


                label = (
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(
            frame,
            f"IBVAP | {CAMERA_ID}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confirmed Events: {event_counter}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


print()
print("=" * 60)
print("IBVAP VIRTUAL FENCE V3 COMPLETE")
print("=" * 60)
print(f"Resolution   : {width}x{height}")
print(f"Output video : {OUTPUT_VIDEO}")
print(f"Event log    : {EVENT_LOG}")
print(f"Events       : {event_counter}")