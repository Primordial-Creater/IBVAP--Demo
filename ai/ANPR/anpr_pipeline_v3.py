from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import json
import re

import cv2
import easyocr
from ultralytics import YOLO


# ============================================================
# IBVAP ANPR V3
# Vehicle-aware ANPR
# Multi-preprocessing OCR + Temporal Consensus
# ============================================================

VIDEO_PATH = "data/test/test_video_ANPR_2.mp4"

VEHICLE_MODEL_PATH = "yolo11n.pt"
PLATE_MODEL_PATH = "ai/models/license_plate_detector.pt"

DEVICE = 0

VEHICLE_CONF = 0.30
PLATE_CONF = 0.25

FRAME_STRIDE = 2
OCR_INTERVAL = 4

OCR_HISTORY_SIZE = 10
MIN_PLATE_LENGTH = 5
MAX_PLATE_LENGTH = 12

# ============================================================
# OUTPUT
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"runs/ibvap/anpr_v3/anpr_v3_{timestamp}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = OUTPUT_DIR / "ibvap_anpr_v2.mp4"
EVENT_LOG = OUTPUT_DIR / "anpr_events_v2.jsonl"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_plate_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    if len(text) < MIN_PLATE_LENGTH:
        return ""

    if len(text) > MAX_PLATE_LENGTH:
        text = text[:MAX_PLATE_LENGTH]

    return text


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def create_plate_variants(image):

    if image is None or image.size == 0:
        return []

    # Large upscale for tiny plates.
    upscaled = cv2.resize(
        image,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # Contrast enhancement.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Sharpen.
    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0
    )

    # Adaptive threshold.
    threshold = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        upscaled,
        enhanced,
        sharpened,
        threshold
    ]


# ============================================================
# OCR
# ============================================================

def recognize_plate(reader, plate_crop):

    variants = create_plate_variants(
        plate_crop
    )

    if not variants:
        return []


    candidates = []


    for variant in variants:

        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for item in results:

            if len(item) < 3:
                continue

            raw_text = item[1]
            confidence = float(item[2])

            cleaned = clean_plate_text(
                raw_text
            )

            if cleaned:

                candidates.append(
                    (
                        cleaned,
                        confidence
                    )
                )

    return candidates


# ============================================================
# PLATE BOX
# ============================================================

def get_best_plate(plate_result):

    if (
        plate_result.boxes is None
        or len(plate_result.boxes) == 0
    ):
        return None

    boxes = (
        plate_result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confidences = (
        plate_result.boxes.conf
        .cpu()
        .numpy()
    )

    best = int(
        confidences.argmax()
    )

    return (
        boxes[best],
        float(confidences[best])
    )


# ============================================================
# BOX SAFETY
# ============================================================

def clamp_box(
    x1,
    y1,
    x2,
    y2,
    width,
    height
):

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("IBVAP ANPR V3")
print("Multi-preprocessing OCR + Temporal Consensus")
print("=" * 70)

print("[INFO] Loading vehicle detector...")

vehicle_model = YOLO(
    VEHICLE_MODEL_PATH
)

print("[INFO] Vehicle model loaded.")

print("[INFO] Loading plate detector...")

plate_model = YOLO(
    PLATE_MODEL_PATH
)

print("[INFO] Plate model loaded.")

print("[INFO] Loading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=True
)

print("[INFO] EasyOCR GPU mode enabled.")


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
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


# ============================================================
# TEMPORAL MEMORY
# ============================================================

ocr_history = defaultdict(list)

last_ocr_frame = {}

confirmed_plates = {}

event_keys = set()


# ============================================================
# COUNTERS
# ============================================================

frame_number = 0

plate_detections = 0

ocr_attempts = 0

ocr_successes = 0

vehicles_seen = set()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    output_frame = frame.copy()


    if frame_number % FRAME_STRIDE == 0:

        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=VEHICLE_CONF,
            device=DEVICE,
            verbose=False
        )

        result = results[0]


        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

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
                ] * len(boxes)


            class_names = {
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck"
            }


            # ====================================================
            # VEHICLES
            # ====================================================

            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                x1, y1, x2, y2 = clamp_box(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height
                )


                if x2 <= x1 or y2 <= y1:
                    continue


                vehicle_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                if vehicle_crop.size == 0:
                    continue


                track_id = int(
                    track_ids[i]
                )

                vehicle_conf = float(
                    confidences[i]
                )

                class_id = int(
                    classes[i]
                )

                vehicle_type = class_names.get(
                    class_id,
                    "vehicle"
                )


                if track_id >= 0:

                    vehicles_seen.add(
                        track_id
                    )


                # =================================================
                # PLATE DETECTION INSIDE VEHICLE
                # =================================================

                plate_results = plate_model.predict(
                    vehicle_crop,
                    conf=PLATE_CONF,
                    device=DEVICE,
                    verbose=False
                )

                plate_result = plate_results[0]

                best_plate = get_best_plate(
                    plate_result
                )


                if best_plate is not None:

                    plate_box, plate_conf = (
                        best_plate
                    )

                    plate_detections += 1


                    px1, py1, px2, py2 = map(
                        int,
                        plate_box
                    )


                    crop_h, crop_w = (
                        vehicle_crop.shape[:2]
                    )


                    px1, py1, px2, py2 = clamp_box(
                        px1,
                        py1,
                        px2,
                        py2,
                        crop_w,
                        crop_h
                    )


                    if px2 > px1 and py2 > py1:

                        plate_crop = vehicle_crop[
                            py1:py2,
                            px1:px2
                        ]


                        # Map plate box back to full frame.

                        fx1 = x1 + px1
                        fy1 = y1 + py1
                        fx2 = x1 + px2
                        fy2 = y1 + py2


                        cv2.rectangle(
                            output_frame,
                            (fx1, fy1),
                            (fx2, fy2),
                            (0, 255, 0),
                            2
                        )


                        cv2.putText(
                            output_frame,
                            f"PLATE {plate_conf:.2f}",
                            (
                                fx1,
                                max(
                                    fy1 - 5,
                                    20
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            2
                        )


                        # =================================================
                        # OCR
                        # =================================================

                        should_ocr = (
                            track_id >= 0
                            and (
                                track_id
                                not in last_ocr_frame

                                or

                                frame_number
                                - last_ocr_frame[track_id]
                                >= OCR_INTERVAL
                            )
                        )


                        if should_ocr:

                            ocr_attempts += 1

                            last_ocr_frame[
                                track_id
                            ] = frame_number


                            candidates = recognize_plate(
                                reader,
                                plate_crop
                            )


                            if candidates:

                                ocr_successes += 1


                                # Add all OCR candidates.

                                for text, confidence in candidates:

                                    ocr_history[
                                        track_id
                                    ].append(
                                        (
                                            text,
                                            confidence
                                        )
                                    )


                                # Keep history bounded.

                                if len(
                                    ocr_history[track_id]
                                ) > OCR_HISTORY_SIZE:

                                    ocr_history[
                                        track_id
                                    ] = (
                                        ocr_history[
                                            track_id
                                        ][
                                            -OCR_HISTORY_SIZE:
                                        ]
                                    )


                                # =================================================
                                # TEMPORAL CONSENSUS
                                # =================================================

                                text_counts = Counter()

                                confidence_totals = defaultdict(float)

                                for text, confidence in (
                                    ocr_history[
                                        track_id
                                    ]
                                ):

                                    text_counts[text] += 1

                                    confidence_totals[
                                        text
                                    ] += confidence


                                if text_counts:

                                    best_text, count = (
                                        text_counts.most_common(1)[0]
                                    )


                                    average_conf = (
                                        confidence_totals[
                                            best_text
                                        ]
                                        /
                                        count
                                    )


                                    # Require repeated agreement.

                                    if count >= 2:

                                        confirmed_plates[
                                            track_id
                                        ] = best_text


                                        event_key = (
                                            track_id,
                                            best_text
                                        )


                                        if (
                                            event_key
                                            not in event_keys
                                        ):

                                            event_keys.add(
                                                event_key
                                            )


                                            event = {

                                                "event_type":
                                                    "ANPR_DETECTION",

                                                "track_id":
                                                    track_id,

                                                "vehicle_type":
                                                    vehicle_type,

                                                "vehicle_confidence":
                                                    round(
                                                        vehicle_conf,
                                                        4
                                                    ),

                                                "plate_text":
                                                    best_text,

                                                "plate_detection_confidence":
                                                    round(
                                                        plate_conf,
                                                        4
                                                    ),

                                                "ocr_confidence":
                                                    round(
                                                        average_conf,
                                                        4
                                                    ),

                                                "ocr_votes":
                                                    count,

                                                "frame":
                                                    frame_number,

                                                "timestamp_seconds":
                                                    round(
                                                        frame_number / fps,
                                                        3
                                                    )
                                            }


                                            with open(
                                                EVENT_LOG,
                                                "a",
                                                encoding="utf-8"
                                            ) as f:

                                                f.write(
                                                    json.dumps(
                                                        event
                                                    )
                                                    + "\n"
                                                )


                # =================================================
                # VEHICLE BOX
                # =================================================

                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


                label = (
                    f"{vehicle_type} "
                    f"ID:{track_id}"
                )


                if track_id in confirmed_plates:

                    label += (
                        f" | "
                        f"{confirmed_plates[track_id]}"
                    )


                cv2.putText(
                    output_frame,
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


    # ========================================================
    # HEADER
    # ========================================================

    cv2.putText(
        output_frame,
        "IBVAP | ANPR V3",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Frame: {frame_number}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Vehicles: {len(vehicles_seen)}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    writer.write(
        output_frame
    )


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
print("IBVAP ANPR V3 COMPLETE")
print("=" * 70)

print(
    f"Resolution          : {width}x{height}"
)

print(
    f"Frames processed    : {frame_number}"
)

print(
    f"Unique vehicles     : {len(vehicles_seen)}"
)

print(
    f"Plate detections    : {plate_detections}"
)

print(
    f"OCR attempts        : {ocr_attempts}"
)

print(
    f"OCR successes       : {ocr_successes}"
)

print(
    f"Confirmed plates    : {len(confirmed_plates)}"
)

print(
    f"Output video        : {OUTPUT_VIDEO}"
)

print(
    f"Event log           : {EVENT_LOG}"
)

print("=" * 70)