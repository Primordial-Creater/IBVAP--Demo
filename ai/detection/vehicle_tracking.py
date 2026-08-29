import cv2
import json
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# IBVAP - VEHICLE DETECTION + CLASSIFICATION + TRACKING
# ============================================================

VIDEO_PATH = "data/test/test_video5.mp4"
MODEL_PATH = "yolo11n.pt"   

OUTPUT_DIR = Path("runs/ibvap/vehicle_v1")
OUTPUT_VIDEO = OUTPUT_DIR / "ibvap_vehicle_v5.mp4"
EVENT_LOG = OUTPUT_DIR / "vehicle_events_v5.jsonl"

DEVICE = 0
CONFIDENCE = 0.35


# ============================================================
# COCO VEHICLE CLASSES
#
# YOLO class IDs:
# 2  = car
# 3  = motorcycle
# 5  = bus
# 7  = truck
# ============================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


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
# TRACKING STATE
# ============================================================

frame_number = 0
event_counter = 0


# ============================================================
# EVENT LOG
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
        # YOLO DETECTION + TRACKING
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASS_IDS,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]


        # ====================================================
        # PROCESS VEHICLES
        # ====================================================

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
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
                    None
                ] * len(boxes)


            # =================================================
            # EACH VEHICLE
            # =================================================

            for (
                box,
                class_id,
                confidence,
                track_id
            ) in zip(
                boxes,
                class_ids,
                confidences,
                track_ids
            ):

                if class_id not in VEHICLE_CLASSES:
                    continue


                vehicle_type = (
                    VEHICLE_CLASSES[class_id]
                )


                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # =================================================
                # CENTER POINT
                # =================================================

                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )


                # =================================================
                # EVENT DATA
                # =================================================

                event_counter += 1

                event = {
                    "event_type": "VEHICLE_DETECTED",
                    "frame_number": frame_number,
                    "vehicle_type": vehicle_type,
                    "class_id": int(class_id),
                    "track_id": (
                        int(track_id)
                        if track_id is not None
                        else None
                    ),
                    "confidence": round(
                        float(confidence),
                        4
                    ),
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "center": [
                        cx,
                        cy
                    ]
                }


                event_file.write(
                    json.dumps(event) + "\n"
                )


                # =================================================
                # DRAW VEHICLE
                # =================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


                label = (
                    f"{vehicle_type} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2
                )


        # ====================================================
        # SYSTEM STATUS
        # ====================================================

        cv2.putText(
            frame,
            "IBVAP | Vehicle Analytics",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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
print("IBVAP VEHICLE ANALYTICS V1 COMPLETE")
print("=" * 60)
print(f"Resolution   : {width}x{height}")
print(f"Output video : {OUTPUT_VIDEO}")
print(f"Event log    : {EVENT_LOG}")
print(f"Vehicle detections logged: {event_counter}")