import cv2
import json
import re
import math
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque, Counter

import numpy as np
import easyocr
from ultralytics import YOLO


# ============================================================
# IBVAP UNIFIED AI PIPELINE V2
# ============================================================
#
# Modules:
#   1. Person detection + tracking
#   2. Vehicle detection + classification + tracking
#   3. Face detection
#   4. ANPR plate detection + OCR
#   5. Virtual fence intrusion
#   6. Night detection
#   7. Suspicious behavior
#   8. Unified event logging
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_MODEL_PATH = "yolo11m.pt"

FACE_MODEL_PATH = "ai/face/yolov11n-face.pt"

PLATE_MODEL_PATH = "ai/models/license_plate_detector.pt"

DEVICE = 0

PERSON_CLASS = 0

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

ALL_TRACK_CLASSES = [0, 2, 3, 5, 7]

PERSON_CONFIDENCE = 0.35
VEHICLE_CONFIDENCE = 0.30

FACE_CONFIDENCE = 0.35
PLATE_CONFIDENCE = 0.25

TRACKER = "bytetrack.yaml"


# ============================================================
# PERFORMANCE
# ============================================================

DETECTION_INTERVAL = 1

FACE_INTERVAL = 3

PLATE_INTERVAL = 4

OCR_INTERVAL = 8


# ============================================================
# NIGHT DETECTION
# ============================================================

NIGHT_ENTER_BRIGHTNESS = 72
NIGHT_EXIT_BRIGHTNESS = 100

NIGHT_ENTER_DARK_RATIO = 0.55
NIGHT_EXIT_DARK_RATIO = 0.40

NIGHT_ENTER_FRAMES = 15
NIGHT_EXIT_FRAMES = 30

EMA_ALPHA = 0.08


# ============================================================
# BEHAVIOR
# ============================================================

MOVEMENT_WINDOW = 12

RAPID_MOVEMENT_DISTANCE = 45

LOITER_DISTANCE = 35
LOITER_FRAMES = 120

NIGHT_MOVEMENT_DISTANCE = 18

EVENT_COOLDOWN = 100


# ============================================================
# VIRTUAL FENCE
# ============================================================
#
# Normalized coordinates.
# Automatically adapts to video resolution.
#
# Change these later for the actual camera.
#
# ============================================================

FENCE_NORMALIZED = [
    (0.05, 0.55),
    (0.95, 0.55),
    (0.95, 0.95),
    (0.05, 0.95)
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="IBVAP Unified AI Pipeline V2"
)

parser.add_argument(
    "--source",
    required=True,
    help="Video path, webcam index, or RTSP/HTTP stream URL"
)

parser.add_argument(
    "--camera-id",
    default="CAM_001",
    help="Camera identifier"
)

parser.add_argument(
    "--no-face",
    action="store_true",
    help="Disable face detection"
)

parser.add_argument(
    "--no-anpr",
    action="store_true",
    help="Disable ANPR"
)

parser.add_argument(
    "--no-fence",
    action="store_true",
    help="Disable virtual fence"
)

args = parser.parse_args()


# ============================================================
# OUTPUT
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = Path(
    f"runs/ibvap/unified_v2/unified_v2_{timestamp}"
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
# STARTUP
# ============================================================

print("=" * 75)
print("IBVAP UNIFIED AI PIPELINE V2")
print("=" * 75)

print("[INFO] Source:", args.source)
print("[INFO] Camera:", args.camera_id)
print("[INFO] Device: CUDA GPU 0")


# ============================================================
# LOAD MAIN YOLO
# ============================================================

print("[INFO] Loading YOLO11m...")

main_model = YOLO(
    YOLO_MODEL_PATH
)

print("[INFO] YOLO11m loaded.")


# ============================================================
# LOAD FACE MODEL
# ============================================================

face_model = None

if not args.no_face:

    print("[INFO] Loading face detector...")

    face_model = YOLO(
        FACE_MODEL_PATH
    )

    print("[INFO] Face detector loaded.")


# ============================================================
# LOAD PLATE MODEL + OCR
# ============================================================

plate_model = None
ocr_reader = None

if not args.no_anpr:

    print("[INFO] Loading license plate detector...")

    plate_model = YOLO(
        PLATE_MODEL_PATH
    )

    print("[INFO] License plate detector loaded.")

    print("[INFO] Loading EasyOCR...")

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=True
    )

    print("[INFO] EasyOCR GPU mode enabled.")


# ============================================================
# OPEN SOURCE
# ============================================================

source = args.source

try:
    source = int(source)
except ValueError:
    pass

cap = cv2.VideoCapture(source)

if not cap.isOpened():

    raise RuntimeError(
        f"Unable to open source: {args.source}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

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

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(
    f"[INFO] Resolution: {width}x{height}"
)

print(
    f"[INFO] FPS: {fps:.2f}"
)

print(
    f"[INFO] Total frames: {total_frames}"
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
# FENCE
# ============================================================

fence = [
    (
        int(x * width),
        int(y * height)
    )
    for x, y in FENCE_NORMALIZED
]

fence_array = np.array(
    fence,
    dtype=np.int32
)


# ============================================================
# TRACK STATE
# ============================================================

positions = defaultdict(
    lambda: deque(
        maxlen=MOVEMENT_WINDOW
    )
)

first_seen = {}

loiter_counter = defaultdict(int)

last_event_frame = defaultdict(
    lambda: -999999
)

fence_state = {}

seen_persons = set()
seen_vehicles = set()


# ============================================================
# ANPR STATE
# ============================================================

ocr_history = defaultdict(list)

confirmed_plates = {}

last_ocr_frame = defaultdict(
    lambda: -999999
)


# ============================================================
# NIGHT STATE
# ============================================================

night_state = False

night_enter_counter = 0
night_exit_counter = 0

brightness_ema = None

night_frames = 0


# ============================================================
# COUNTERS
# ============================================================

frame_number = 0

event_count = 0

person_detections = 0
vehicle_detections = 0
face_detections = 0
plate_detections = 0

anpr_events = 0
intrusion_events = 0
night_events = 0
behavior_events = 0


# ============================================================
# EVENT LOGGER
# ============================================================

event_file = open(
    EVENT_LOG,
    "w",
    encoding="utf-8"
)


def write_event(
    event_type,
    track_id=None,
    confidence=None,
    data=None,
    severity="INFO"
):

    global event_count

    event_count += 1

    event = {

        "event_id":
            f"EVT_{event_count:06d}",

        "event_type":
            event_type,

        "camera_id":
            args.camera_id,

        "track_id":
            int(track_id)
            if track_id is not None
            else None,

        "confidence":
            round(float(confidence), 4)
            if confidence is not None
            else None,

        "frame_number":
            int(frame_number),

        "timestamp_seconds":
            round(
                frame_number / fps,
                3
            ),

        "severity":
            severity,

        "data":
            data or {}

    }

    event_file.write(
        json.dumps(event) + "\n"
    )

    event_file.flush()

    return event


# ============================================================
# NIGHT METRICS
# ============================================================

def calculate_night_metrics(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    valid_pixels = gray[
        (gray > 5) &
        (gray < 245)
    ]

    if len(valid_pixels) == 0:

        return 0.0, 1.0

    median_brightness = float(
        np.median(valid_pixels)
    )

    dark_ratio = float(
        np.sum(valid_pixels < 80)
        /
        len(valid_pixels)
    )

    return (
        median_brightness,
        dark_ratio
    )


# ============================================================
# POINT INSIDE FENCE
# ============================================================

def inside_fence(point):

    return (
        cv2.pointPolygonTest(
            fence_array,
            (
                float(point[0]),
                float(point[1])
            ),
            False
        )
        >= 0
    )


# ============================================================
# PLATE TEXT CLEANING
# ============================================================

def clean_plate_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    if len(text) < 5:
        return ""

    if len(text) > 12:
        text = text[:12]

    return text


# ============================================================
# PLATE PREPROCESSING
# ============================================================

def create_plate_variants(image):

    if image is None:
        return []

    if image.size == 0:
        return []

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

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

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

def recognize_plate(
    plate_crop
):

    candidates = []

    variants = create_plate_variants(
        plate_crop
    )

    for variant in variants:

        results = ocr_reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for item in results:

            if len(item) < 3:
                continue

            raw_text = item[1]

            confidence = float(
                item[2]
            )

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
# BEST PLATE
# ============================================================

def get_best_plate(result):

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

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

    best = int(
        confidences.argmax()
    )

    return (
        boxes[best],
        float(confidences[best])
    )


# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    output_frame = frame.copy()

    timestamp_seconds = (
        frame_number / fps
    )


    # ========================================================
    # NIGHT ANALYSIS
    # ========================================================

    brightness, dark_ratio = (
        calculate_night_metrics(
            frame
        )
    )

    if brightness_ema is None:

        brightness_ema = brightness

    else:

        brightness_ema = (
            EMA_ALPHA * brightness
            +
            (1 - EMA_ALPHA)
            * brightness_ema
        )


    night_candidate = (
        brightness_ema
        < NIGHT_ENTER_BRIGHTNESS
        and
        dark_ratio
        > NIGHT_ENTER_DARK_RATIO
    )

    day_candidate = (
        brightness_ema
        > NIGHT_EXIT_BRIGHTNESS
        or
        dark_ratio
        < NIGHT_EXIT_DARK_RATIO
    )


    if not night_state:

        if night_candidate:

            night_enter_counter += 1

        else:

            night_enter_counter = 0

        if (
            night_enter_counter
            >= NIGHT_ENTER_FRAMES
        ):

            night_state = True

            night_enter_counter = 0

            write_event(
                "NIGHT_MODE_ENTERED",
                data={
                    "brightness":
                        round(
                            brightness_ema,
                            2
                        ),
                    "dark_ratio":
                        round(
                            dark_ratio,
                            3
                        )
                }
            )

    else:

        if day_candidate:

            night_exit_counter += 1

        else:

            night_exit_counter = 0

        if (
            night_exit_counter
            >= NIGHT_EXIT_FRAMES
        ):

            night_state = False

            night_exit_counter = 0

            write_event(
                "DAY_MODE_ENTERED",
                data={
                    "brightness":
                        round(
                            brightness_ema,
                            2
                        ),
                    "dark_ratio":
                        round(
                            dark_ratio,
                            3
                        )
                }
            )


    if night_state:

        night_frames += 1


    # ========================================================
    # MAIN PERSON + VEHICLE TRACKING
    # ========================================================

    results = main_model.track(

        frame,

        persist=True,

        tracker=TRACKER,

        classes=ALL_TRACK_CLASSES,

        conf=VEHICLE_CONFIDENCE,

        device=DEVICE,

        verbose=False

    )

    result = results[0]


    persons = []
    vehicles = []


    if (
        result.boxes is not None
        and
        len(result.boxes) > 0
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


        # ====================================================
        # PROCESS OBJECTS
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

            if x2 <= x1 or y2 <= y1:
                continue


            track_id = int(
                track_id
            )

            confidence = float(
                confidence
            )


            # =================================================
            # PERSON
            # =================================================

            if class_id == PERSON_CLASS:

                person_detections += 1

                persons.append(
                    {
                        "track_id":
                            track_id,

                        "confidence":
                            confidence,

                        "bbox":
                            (x1, y1, x2, y2)
                    }
                )


                if track_id >= 0:

                    seen_persons.add(
                        track_id
                    )

                    cx = (
                        x1 + x2
                    ) // 2

                    # FOOT POINT is better
                    # for fence intrusion
                    cy = y2

                    positions[
                        track_id
                    ].append(
                        (
                            cx,
                            cy
                        )
                    )


                    if (
                        track_id
                        not in first_seen
                    ):

                        first_seen[
                            track_id
                        ] = frame_number

                        write_event(
                            "PERSON_DETECTED",

                            track_id,

                            confidence,

                            {
                                "bbox": [
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ]
                            }
                        )


                    # =========================================
                    # MOVEMENT
                    # =========================================

                    movement = 0.0

                    history = positions[
                        track_id
                    ]

                    if len(history) >= 2:

                        old_x, old_y = (
                            history[0]
                        )

                        movement = math.sqrt(
                            (
                                cx - old_x
                            ) ** 2
                            +
                            (
                                cy - old_y
                            ) ** 2
                        )


                    # =========================================
                    # RAPID MOVEMENT
                    # =========================================

                    if (

                        movement
                        >=
                        RAPID_MOVEMENT_DISTANCE

                        and

                        frame_number
                        -
                        last_event_frame[
                            track_id
                        ]
                        >
                        EVENT_COOLDOWN

                    ):

                        write_event(

                            "RAPID_MOVEMENT",

                            track_id,

                            confidence,

                            {
                                "movement_pixels":
                                    round(
                                        movement,
                                        2
                                    )
                            },

                            "MEDIUM"

                        )

                        behavior_events += 1

                        last_event_frame[
                            track_id
                        ] = frame_number


                    # =========================================
                    # LOITERING
                    # =========================================

                    if (
                        movement
                        <
                        LOITER_DISTANCE
                    ):

                        loiter_counter[
                            track_id
                        ] += 1

                    else:

                        loiter_counter[
                            track_id
                        ] = 0


                    if (
                        loiter_counter[
                            track_id
                        ]
                        >=
                        LOITER_FRAMES

                        and

                        frame_number
                        -
                        last_event_frame[
                            track_id
                        ]
                        >
                        EVENT_COOLDOWN
                    ):

                        duration = (
                            frame_number
                            -
                            first_seen[
                                track_id
                            ]
                        ) / fps

                        write_event(

                            "LOITERING",

                            track_id,

                            confidence,

                            {
                                "duration_seconds":
                                    round(
                                        duration,
                                        2
                                    ),

                                "movement_pixels":
                                    round(
                                        movement,
                                        2
                                    )
                            },

                            "MEDIUM"

                        )

                        behavior_events += 1

                        last_event_frame[
                            track_id
                        ] = frame_number


                    # =========================================
                    # NIGHT MOVEMENT
                    # =========================================

                    if (

                        night_state

                        and

                        movement
                        >=
                        NIGHT_MOVEMENT_DISTANCE

                        and

                        frame_number
                        -
                        last_event_frame[
                            track_id
                        ]
                        >
                        EVENT_COOLDOWN

                    ):

                        write_event(

                            "NIGHT_TIME_MOVEMENT",

                            track_id,

                            confidence,

                            {
                                "movement_pixels":
                                    round(
                                        movement,
                                        2
                                    ),

                                "brightness":
                                    round(
                                        brightness_ema,
                                        2
                                    )
                            },

                            "HIGH"

                        )

                        night_events += 1

                        last_event_frame[
                            track_id
                        ] = frame_number


                    # =========================================
                    # VIRTUAL FENCE
                    # =========================================

                    if not args.no_fence:

                        inside = inside_fence(
                            (cx, cy)
                        )

                        previous = fence_state.get(
                            track_id,
                            False
                        )

                        fence_state[
                            track_id
                        ] = inside


                        # ENTER EVENT
                        if (
                            inside
                            and not previous
                        ):

                            write_event(

                                "VIRTUAL_FENCE_INTRUSION",

                                track_id,

                                confidence,

                                {
                                    "position": {
                                        "x": cx,
                                        "y": cy
                                    },

                                    "bbox": [
                                        x1,
                                        y1,
                                        x2,
                                        y2
                                    ]
                                },

                                "HIGH"

                            )

                            intrusion_events += 1


                        # EXIT EVENT
                        elif (
                            not inside
                            and previous
                        ):

                            write_event(

                                "VIRTUAL_FENCE_EXIT",

                                track_id,

                                confidence,

                                {
                                    "position": {
                                        "x": cx,
                                        "y": cy
                                    }
                                },

                                "INFO"

                            )


                    # =========================================
                    # DRAW PERSON
                    # =========================================

                    if night_state:

                        box_color = (
                            255,
                            0,
                            255
                        )

                    else:

                        box_color = (
                            0,
                            255,
                            0
                        )


                    cv2.rectangle(

                        output_frame,

                        (x1, y1),

                        (x2, y2),

                        box_color,

                        2

                    )


                    cv2.putText(

                        output_frame,

                        f"PERSON ID:{track_id} "
                        f"{confidence:.2f}",

                        (
                            x1,
                            max(
                                20,
                                y1 - 8
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.5,

                        box_color,

                        2

                    )


            # =================================================
            # VEHICLE
            # =================================================

            elif class_id in VEHICLE_CLASSES:

                vehicle_detections += 1

                vehicle_type = (
                    VEHICLE_CLASSES[
                        class_id
                    ]
                )

                vehicles.append(
                    {
                        "track_id":
                            track_id,

                        "confidence":
                            confidence,

                        "bbox":
                            (x1, y1, x2, y2),

                        "vehicle_type":
                            vehicle_type
                    }
                )

                if track_id >= 0:

                    seen_vehicles.add(
                        track_id
                    )


                # =============================================
                # DRAW VEHICLE
                # =============================================

                cv2.rectangle(

                    output_frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 0, 0),

                    2

                )

                label = (

                    f"{vehicle_type.upper()} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"

                )


                cv2.putText(

                    output_frame,

                    label,

                    (
                        x1,
                        max(
                            20,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 0, 0),

                    2

                )


    # ========================================================
    # FACE DETECTION
    # ========================================================

    if (
        face_model is not None
        and
        frame_number % FACE_INTERVAL == 0
    ):

        face_results = face_model.predict(

            source=frame,

            conf=FACE_CONFIDENCE,

            imgsz=640,

            device=DEVICE,

            verbose=False

        )

        face_result = (
            face_results[0]
        )


        if (
            face_result.boxes is not None
        ):

            for face_box in (
                face_result.boxes
            ):

                coords = (
                    face_box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                fx1, fy1, fx2, fy2 = (
                    coords
                )

                fconf = float(
                    face_box.conf[0]
                    .cpu()
                )

                face_detections += 1


                cv2.rectangle(

                    output_frame,

                    (fx1, fy1),

                    (fx2, fy2),

                    (0, 0, 255),

                    2

                )

                cv2.putText(

                    output_frame,

                    f"FACE {fconf:.2f}",

                    (
                        fx1,
                        max(
                            20,
                            fy1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0, 0, 255),

                    2

                )


                write_event(

                    "FACE_DETECTED",

                    confidence=fconf,

                    data={
                        "bbox": [
                            int(fx1),
                            int(fy1),
                            int(fx2),
                            int(fy2)
                        ],

                        "recognition_status":
                            "PENDING_DATABASE_LOOKUP"
                    }

                )


    # ========================================================
    # ANPR
    # ========================================================

    if (

        plate_model is not None

        and

        frame_number % PLATE_INTERVAL == 0

    ):

        for vehicle in vehicles:

            x1, y1, x2, y2 = (
                vehicle["bbox"]
            )

            vehicle_crop = frame[
                y1:y2,
                x1:x2
            ]


            if vehicle_crop.size == 0:
                continue


            plate_results = (
                plate_model.predict(

                    source=vehicle_crop,

                    conf=PLATE_CONFIDENCE,

                    device=DEVICE,

                    verbose=False

                )
            )


            plate_result = (
                plate_results[0]
            )


            best_plate = (
                get_best_plate(
                    plate_result
                )
            )


            if best_plate is None:
                continue


            plate_box, plate_conf = (
                best_plate
            )

            plate_detections += 1


            px1, py1, px2, py2 = map(
                int,
                plate_box
            )


            # Convert vehicle-local
            # coordinates to frame coordinates

            px1 += x1
            px2 += x1

            py1 += y1
            py2 += y1


            cv2.rectangle(

                output_frame,

                (px1, py1),

                (px2, py2),

                (0, 255, 255),

                2

            )


            # =============================================
            # OCR
            # =============================================

            if (

                frame_number
                -
                last_ocr_frame[
                    vehicle["track_id"]
                ]
                >=
                OCR_INTERVAL

            ):

                last_ocr_frame[
                    vehicle["track_id"]
                ] = frame_number


                crop = frame[
                    max(0, py1):
                    min(height, py2),

                    max(0, px1):
                    min(width, px2)
                ]


                candidates = recognize_plate(
                    crop
                )


                for text, confidence in candidates:

                    ocr_history[
                        vehicle["track_id"]
                    ].append(
                        (
                            text,
                            confidence
                        )
                    )


                history = ocr_history[
                    vehicle["track_id"]
                ]


                if history:

                    counts = Counter(
                        text
                        for text, _ in history
                    )

                    best_text, votes = (
                        counts.most_common(1)[0]
                    )


                    confidence_values = [

                        conf

                        for text, conf
                        in history

                        if text == best_text

                    ]


                    average_conf = (
                        sum(
                            confidence_values
                        )
                        /
                        len(
                            confidence_values
                        )
                    )


                    # Require repeated agreement
                    if votes >= 2:

                        previous_plate = (
                            confirmed_plates.get(
                                vehicle["track_id"]
                            )
                        )

                        confirmed_plates[
                            vehicle["track_id"]
                        ] = best_text


                        if (
                            previous_plate
                            !=
                            best_text
                        ):

                            write_event(

                                "ANPR_DETECTION",

                                vehicle[
                                    "track_id"
                                ],

                                vehicle[
                                    "confidence"
                                ],

                                {
                                    "vehicle_type":
                                        vehicle[
                                            "vehicle_type"
                                        ],

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
                                        votes
                                },

                                "HIGH"

                            )

                            anpr_events += 1


            # =============================================
            # SHOW CONFIRMED PLATE
            # =============================================

            plate_text = (
                confirmed_plates.get(
                    vehicle["track_id"]
                )
            )

            if plate_text:

                cv2.putText(

                    output_frame,

                    plate_text,

                    (
                        px1,
                        max(
                            20,
                            py1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 255),

                    2

                )


    # ========================================================
    # DRAW FENCE
    # ========================================================

    if not args.no_fence:

        cv2.polylines(

            output_frame,

            [fence_array],

            True,

            (0, 165, 255),

            2

        )

        # cv2.putText(

        #     output_frame,

        #     "RESTRICTED ZONE",

        #     (
        #         fence[0][0],
        #         max(
        #             20,
        #             fence[0][1] - 10
        #         )
        #     ),

        #     cv2.FONT_HERSHEY_SIMPLEX,

        #     0.6,

        #     (0, 165, 255),

        #     2

        # )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    mode = (
        "NIGHT"
        if night_state
        else "DAY"
    )


    cv2.rectangle(

        output_frame,

        (10, 10),

        (430, 125),

        (0, 0, 0),

        -1

    )


    cv2.putText(

        output_frame,

        "IBVAP | UNIFIED AI V2",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    cv2.putText(

        output_frame,

        f"MODE: {mode}",

        (20, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1

    )


    cv2.putText(

        output_frame,

        f"Persons: {len(persons)} | "
        f"Vehicles: {len(vehicles)}",

        (20, 83),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1

    )


    cv2.putText(

        output_frame,

        f"Faces: {face_detections} | "
        f"ANPR: {anpr_events}",

        (20, 104),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        (255, 255, 255),

        1

    )


    # ========================================================
    # WRITE
    # ========================================================

    writer.write(
        output_frame
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

event_file.close()


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


print()

print("=" * 75)
print("IBVAP UNIFIED AI PIPELINE V2 COMPLETE")
print("=" * 75)

print(
    f"Resolution        : "
    f"{width}x{height}"
)

print(
    f"Frames processed  : "
    f"{frame_number}"
)

print(
    f"Unique persons    : "
    f"{len(seen_persons)}"
)

print(
    f"Unique vehicles   : "
    f"{len(seen_vehicles)}"
)

print(
    f"Face detections   : "
    f"{face_detections}"
)

print(
    f"Plate detections  : "
    f"{plate_detections}"
)

print(
    f"ANPR events       : "
    f"{anpr_events}"
)

print(
    f"Fence intrusions  : "
    f"{intrusion_events}"
)

print(
    f"Night events      : "
    f"{night_events}"
)

print(
    f"Behavior events   : "
    f"{behavior_events}"
)

print(
    f"Total events      : "
    f"{event_count}"
)

print(
    f"Processing time   : "
    f"{elapsed:.2f} seconds"
)

print(
    f"Output video      : "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Event log         : "
    f"{EVENT_LOG}"
)

print("=" * 75)