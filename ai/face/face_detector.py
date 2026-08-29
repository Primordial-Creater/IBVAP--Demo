import cv2
import json
import time
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO


# ============================================================
# IBVAP - FACE DETECTION V1
# ============================================================

MODEL_PATH = "ai/face/yolov11n-face.pt"

INPUT_VIDEO = "data/test/test _face_detector4.mp4"

OUTPUT_DIR = Path("runs/ibvap/face")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_VIDEO = OUTPUT_DIR / f"ibvap_face_v1_{timestamp}.mp4"
EVENT_FILE = OUTPUT_DIR / f"face_events_v1_{timestamp}.jsonl"


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.35
IMG_SIZE = 640


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 70)
print("IBVAP FACE DETECTION V1")
print("=" * 70)

print("[INFO] Loading face detection model...")

model = YOLO(MODEL_PATH)

print("[INFO] Face model loaded.")

# Explicit GPU
DEVICE = 0

print("[INFO] Using GPU:", DEVICE)


# ------------------------------------------------------------
# OPEN VIDEO
# ------------------------------------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print(f"[INFO] Resolution: {width}x{height}")
print(f"[INFO] FPS: {fps:.2f}")
print(f"[INFO] Frames: {total_frames}")


# ------------------------------------------------------------
# VIDEO WRITER
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ------------------------------------------------------------
# PROCESS VIDEO
# ------------------------------------------------------------

frame_number = 0
face_count = 0

start_time = time.time()

with open(EVENT_FILE, "w", encoding="utf-8") as event_log:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for detection_index, box in enumerate(boxes):

                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                x1, y1, x2, y2 = xyxy

                confidence = float(box.conf[0].cpu())

                face_count += 1

                # ------------------------------------------------
                # DRAW DETECTION
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                label = f"FACE {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # ------------------------------------------------
                # EVENT
                # ------------------------------------------------

                event = {
                    "event_type": "FACE_DETECTION",
                    "frame": frame_number,
                    "timestamp_seconds": round(
                        frame_number / fps,
                        3
                    ),
                    "face_index": detection_index,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                }

                event_log.write(
                    json.dumps(event) + "\n"
                )

        # --------------------------------------------------------
        # STATUS OVERLAY
        # --------------------------------------------------------

        cv2.putText(
            frame,
            f"IBVAP | FACE DETECTION | Frame {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()
writer.release()

elapsed = time.time() - start_time

print()
print("=" * 70)
print("IBVAP FACE DETECTION V1 COMPLETE")
print("=" * 70)

print(f"Resolution      : {width}x{height}")
print(f"Frames processed: {frame_number}")
print(f"Face detections : {face_count}")
print(f"Processing time : {elapsed:.2f} seconds")
print(f"Output video    : {OUTPUT_VIDEO}")
print(f"Event log       : {EVENT_FILE}")

print("=" * 70)