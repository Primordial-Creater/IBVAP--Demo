import cv2
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


# ============================================================
# IBVAP UNIFIED AI PIPELINE V2
# Source-configurable architecture
# ============================================================

MODEL_PATH = "yolo11m.pt"

DEVICE = 0
CONFIDENCE = 0.35

CAMERA_ID = "CAM_001"


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="IBVAP Unified AI Pipeline"
)

parser.add_argument(
    "--source",
    required=True,
    help="Video path, webcam index, or RTSP URL"
)

parser.add_argument(
    "--camera-id",
    default=CAMERA_ID,
    help="Camera identifier"
)

args = parser.parse_args()

SOURCE = args.source
CAMERA_ID = args.camera_id


# ============================================================
# OUTPUT
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"runs/ibvap/unified/unified_v2_{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = (
    OUTPUT_DIR /
    "ibvap_unified_v2.mp4"
)

EVENT_LOG = (
    OUTPUT_DIR /
    "ibvap_unified_events_v2.jsonl"
)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("IBVAP UNIFIED AI PIPELINE V2")
print("=" * 75)

print(f"[INFO] Source     : {SOURCE}")
print(f"[INFO] Camera ID  : {CAMERA_ID}")
print(f"[INFO] Device     : CUDA:{DEVICE}")
print("[INFO] Loading YOLO11m...")


# ============================================================
# MODEL
# ============================================================

model = YOLO(MODEL_PATH)

print("[INFO] YOLO11m loaded.")


# ============================================================
# SOURCE HANDLING
# ============================================================

source = SOURCE

if source.isdigit():
    source = int(source)

cap = cv2.VideoCapture(source)

if not cap.isOpened():

    raise RuntimeError(
        f"Unable to open source: {SOURCE}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACK STATE
# ============================================================

seen_person_tracks = set()
seen_vehicle_tracks = set()

person_count = 0
vehicle_count = 0

frame_number = 0

start_time = time.time()


# ============================================================
# EVENT LOGGER
# ============================================================

event_count = 0


def emit_event(
    event_type,
    track_id,
    confidence,
    frame_number,
    timestamp_seconds,
    data=None,
    severity="INFO"
):

    global event_count

    event = {

        "event_type": event_type,

        "camera_id": CAMERA_ID,

        "track_id": track_id,

        "confidence": round(
            float(confidence),
            4
        ),

        "frame_number": frame_number,

        "timestamp_seconds": round(
            float(timestamp_seconds),
            3
        ),

        "severity": severity,

        "data": data or {}
    }

    with open(
        EVENT_LOG,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(event)
            + "\n"
        )

    event_count += 1


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    timestamp_seconds = (
        frame_number / fps
    )


    # ========================================================
    # PERSON + VEHICLE DETECTION/TRACKING
    # ========================================================

    results = model.track(

        frame,

        persist=True,

        tracker="bytetrack.yaml",

        conf=CONFIDENCE,

        device=DEVICE,

        verbose=False
    )

    result = results[0]


    current_persons = 0
    current_vehicles = 0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        if result.boxes.id is not None:

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

        else:

            track_ids = [
                -1
                for _ in boxes
            ]


        # ====================================================
        # EACH DETECTION
        # ====================================================

        for (
            box,
            class_id,
            confidence,
            track_id
        ) in zip(
            boxes,
            classes,
            confidences,
            track_ids
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            track_id = int(track_id)

            confidence = float(
                confidence
            )


            # =================================================
            # PERSON
            # =================================================

            if class_id == 0:

                current_persons += 1

                if (
                    track_id >= 0
                    and
                    track_id not in seen_person_tracks
                ):

                    seen_person_tracks.add(
                        track_id
                    )

                    person_count += 1

                    emit_event(

                        "PERSON_DETECTED",

                        track_id,

                        confidence,

                        frame_number,

                        timestamp_seconds,

                        {
                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2
                            ]
                        }
                    )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )


                cv2.putText(

                    frame,

                    f"PERSON ID:{track_id} "
                    f"{confidence:.2f}",

                    (
                        x1,
                        max(
                            25,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0, 255, 0),

                    2
                )


            # =================================================
            # VEHICLES
            # =================================================

            elif class_id in {

                2,   # car
                3,   # motorcycle
                5,   # bus
                7    # truck

            }:

                current_vehicles += 1

                if (
                    track_id >= 0
                    and
                    track_id not in seen_vehicle_tracks
                ):

                    seen_vehicle_tracks.add(
                        track_id
                    )

                    vehicle_count += 1

                    vehicle_names = {

                        2: "car",

                        3: "motorcycle",

                        5: "bus",

                        7: "truck"
                    }

                    vehicle_type = (
                        vehicle_names[class_id]
                    )

                    emit_event(

                        "VEHICLE_DETECTED",

                        track_id,

                        confidence,

                        frame_number,

                        timestamp_seconds,

                        {

                            "vehicle_type":
                                vehicle_type,

                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2
                            ]
                        }
                    )


                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 0, 0),

                    2
                )


                vehicle_name = {

                    2: "CAR",

                    3: "MOTORCYCLE",

                    5: "BUS",

                    7: "TRUCK"

                }.get(
                    class_id,
                    "VEHICLE"
                )


                cv2.putText(

                    frame,

                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}",

                    (
                        x1,
                        max(
                            25,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 0, 0),

                    2
                )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    cv2.rectangle(

        frame,

        (10, 10),

        (470, 125),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        frame,

        "IBVAP | UNIFIED AI PIPELINE V2",

        (20, 38),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        f"Camera: {CAMERA_ID}",

        (20, 63),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Persons: {current_persons} | "
        f"Vehicles: {current_vehicles}",

        (20, 88),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    cv2.putText(

        frame,

        f"Unique P: {person_count} | "
        f"Unique V: {vehicle_count}",

        (20, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1
    )


    # ========================================================
    # WRITE
    # ========================================================

    writer.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()


elapsed = (
    time.time() - start_time
)


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 75)

print(
    "IBVAP UNIFIED AI PIPELINE V2 COMPLETE"
)

print("=" * 75)

print(
    f"Resolution       : "
    f"{width}x{height}"
)

print(
    f"Frames processed : "
    f"{frame_number}"
)

print(
    f"Unique persons   : "
    f"{person_count}"
)

print(
    f"Unique vehicles  : "
    f"{vehicle_count}"
)

print(
    f"Events generated : "
    f"{event_count}"
)

print(
    f"Processing time  : "
    f"{elapsed:.2f} seconds"
)

print(
    f"Output video     : "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Event log        : "
    f"{EVENT_LOG}"
)

print("=" * 75)