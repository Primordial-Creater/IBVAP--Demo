# from datetime import datetime
# from typing import Any


# class EventEngine:

#     def __init__(self):
#         self.events = []

#     def create_event(
#         self,
#         event_type: str,
#         frame: int,
#         timestamp_seconds: float,
#         data: dict[str, Any] | None = None,
#     ) -> dict[str, Any]:

#         event = {
#             "event_id": len(self.events) + 1,
#             "event_type": event_type,
#             "frame": frame,
#             "timestamp_seconds": round(timestamp_seconds, 3),
#             "timestamp_utc": datetime.utcnow().isoformat(),
#             "data": data or {},
#         }

#         self.events.append(event)

#         return event

#     def add_event(self, event: dict[str, Any]):
#         self.events.append(event)

#     def get_events(self):
#         return self.events

#     def clear(self):
#         self.events.clear()



































import json
from pathlib import Path
from datetime import datetime


class EventEngine:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.event_count = 0

        self.file = open(
            self.output_path,
            "a",
            encoding="utf-8"
        )

    def emit(
        self,
        event_type,
        camera_id="CAM_001",
        track_id=None,
        confidence=None,
        frame_number=None,
        timestamp_seconds=None,
        data=None,
        severity="INFO"
    ):
        self.event_count += 1

        event = {
            "event_id": f"EVT_{self.event_count:06d}",
            "event_type": event_type,
            "severity": severity,
            "camera_id": camera_id,
            "track_id": track_id,
            "confidence": confidence,
            "frame_number": frame_number,
            "timestamp_seconds": timestamp_seconds,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "data": data or {}
        }

        self.file.write(
            json.dumps(
                event,
                ensure_ascii=False
            ) + "\n"
        )

        self.file.flush()

        return event

    def close(self):
        self.file.close()