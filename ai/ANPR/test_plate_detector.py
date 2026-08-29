from pathlib import Path
import cv2
from ultralytics import YOLO


# ============================================================
# IBVAP - LICENSE PLATE DETECTOR TEST
# ============================================================

VIDEO_PATH = "data/test/test_video_ANPR_4.mp4"
MODEL_PATH = "ai/models/license_plate_detector.pt"

OUTPUT_DIR = Path("runs/ibvap/anpr_v1")
OUTPUT_VIDEO = OUTPUT_DIR / "anpr_plate_detector_v4.mp4"

CONFIDENCE = 0.30
DEVICE = 0


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 60)
print("IBVAP ANPR - LICENSE PLATE DETECTOR TEST")
print("=" * 60)

print("[INFO] Loading plate detector...")

model = YOLO(MODEL_PATH)

print("[INFO] Model loaded.")


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


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


frame_number = 0
plate_count = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1


    results = model.predict(
        frame,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]


    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )


        for box, confidence in zip(
            boxes,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )

            plate_count += 1


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            label = (
                f"PLATE "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )


    cv2.putText(
        frame,
        "IBVAP | ANPR Plate Detection V1",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    writer.write(frame)


cap.release()
writer.release()


print()
print("=" * 60)
print("ANPR PLATE DETECTOR V1 COMPLETE")
print("=" * 60)
print(f"Resolution      : {width}x{height}")
print(f"Frames processed: {frame_number}")
print(f"Plate detections: {plate_count}")
print(f"Output video    : {OUTPUT_VIDEO}") 