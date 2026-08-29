# class UnifiedAIPipeline:

#     def __init__(self):
#         self.person_model = ...
#         self.vehicle_model = ...
#         self.face_model = ...
#         self.anpr_model = ...
#         self.behavior_model = ...
#         self.event_engine = EventEngine()

#     def process_frame(self, frame, frame_number, timestamp):

#         person_events = self.process_person(frame)

#         vehicle_events = self.process_vehicle(frame)

#         face_events = self.process_face(frame)

#         anpr_events = self.process_anpr(
#             frame,
#             vehicle_events
#         )

#         behavior_events = self.process_behavior(
#             frame,
#             person_events
#         )

#         intrusion_events = self.process_intrusion(
#             person_events
#         )

#         night_events = self.process_night(
#             frame,
#             person_events
#         )

#         return {
#             "persons": person_events,
#             "vehicles": vehicle_events,
#             "faces": face_events,
#             "anpr": anpr_events,
#             "behavior": behavior_events,
#             "intrusion": intrusion_events,
#             "night": night_events,
#         }




































import cv2
import json
import time
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from ai.pipeline.event_engine import EventEngine


# ============================================================
# IBVAP UNIFIED AI PIPELINE V1
# ============================================================

MODEL_PATH = "yolo11m.pt"

DEVICE = 0

PERSON_CLASS = 0

CONFIDENCE = 0.35

CAMERA_ID = "CAM_001"


# ============================================================
# INPUT
# ============================================================

INPUT_VIDEO = "data/test/pipeline_test.WEBM"


# ============================================================
# OUTPUT
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"runs/ibvap/unified/unified_v1_{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = (
    OUTPUT_DIR /
    "ibvap_unified_v1.mp4"
)

EVENT_LOG = (
    OUTPUT_DIR /
    "ibvap_unified_events_v1.jsonl"
)


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 75)
print("IBVAP UNIFIED AI PIPELINE V1")
print("=" * 75)

print("[INFO] Loading YOLO11m...")

model = YOLO(MODEL_PATH)

print("[INFO] YOLO11m loaded.")

print("[INFO] CUDA device:", DEVICE)


# ============================================================
# EVENT ENGINE
# ============================================================

event_engine = EventEngine(
    EVENT_LOG
)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():
    raise RuntimeError(
        f"Unable to open video: {INPUT_VIDEO}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

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

seen_tracks = set()

person_count = 0

frame_number = 0

start_time = time.time()


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
    # PERSON DETECTION + TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS],
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]


    current_persons = 0


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

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

            current_persons += 1

            track_id = int(track_id)

            confidence = float(
                confidence
            )


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # NEW TRACK
            # =================================================

            if track_id not in seen_tracks:

                seen_tracks.add(
                    track_id
                )

                person_count += 1

                event_engine.emit(
                    event_type="PERSON_DETECTED",
                    camera_id=CAMERA_ID,
                    track_id=track_id,
                    confidence=round(
                        confidence,
                        4
                    ),
                    frame_number=frame_number,
                    timestamp_seconds=round(
                        timestamp_seconds,
                        3
                    ),
                    data={
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    },
                    severity="INFO"
                )


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

            label = (
                f"PERSON "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (400, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "IBVAP | UNIFIED AI PIPELINE V1",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Camera: {CAMERA_ID}",
        (20, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"Persons: {current_persons} | Tracks: {person_count}",
        (20, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )


    writer.write(
        frame
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

event_engine.close()


elapsed = (
    time.time() - start_time
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("IBVAP UNIFIED AI PIPELINE V1 COMPLETE")
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
    f"Unique tracks    : "
    f"{len(seen_tracks)}"
)

print(
    f"Events generated : "
    f"{event_engine.event_count}"
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