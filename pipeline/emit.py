# PROMPT: Create an event emission module with schema validation, idempotent
# deduplication by event_id, JSONL output, and emission statistics.
#
# CHANGES MADE: Changed deduplication from content-hash (which caused two identical
# valid events to be wrongly dropped) to event_id dedup as spec requires.
# Confidence validation already correct (0-1). Added partial-batch success support.

import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "event_id", "store_id", "camera_id", "visitor_id",
    "event_type", "timestamp", "is_staff", "confidence",
]
VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
}


class EventEmitter:

    def __init__(self):
        self.emitted_events: List[dict] = []
        self._seen_ids: set = set()
        self.stats = {"accepted": 0, "duplicate": 0, "invalid": 0}

    def validate_event(self, event: dict) -> bool:
        for field in REQUIRED_FIELDS:
            if field not in event:
                logger.warning(f"Missing field: {field}")
                return False
        if event.get("event_type") not in VALID_EVENT_TYPES:
            logger.warning(f"Unknown event_type: {event.get('event_type')}")
            return False
        conf = event.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            logger.warning(f"Confidence out of range: {conf}")
            return False
        try:
            datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"Invalid timestamp: {event.get('timestamp')}")
            return False
        return True

    def emit_event(self, event: dict) -> bool:
        if not self.validate_event(event):
            self.stats["invalid"] += 1
            return False
        eid = event["event_id"]
        if eid in self._seen_ids:          # idempotent by event_id
            self.stats["duplicate"] += 1
            return False
        self._seen_ids.add(eid)
        self.emitted_events.append(event)
        self.stats["accepted"] += 1
        return True

    def emit_batch(self, events: List[dict]) -> int:
        accepted = sum(1 for e in events if self.emit_event(e))
        logger.info(f"Batch: {accepted}/{len(events)} accepted")
        return accepted

    def save_to_jsonl(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for event in self.emitted_events:
                f.write(json.dumps(event) + "\n")
        logger.info(f"Saved {len(self.emitted_events)} events → {output_path}")

    def get_events(self) -> List[dict]:
        return self.emitted_events

    def get_stats(self) -> dict:
        types: dict = {}
        staff_count = 0
        for e in self.emitted_events:
            t = e.get("event_type", "UNKNOWN")
            types[t] = types.get(t, 0) + 1
            if e.get("is_staff"):
                staff_count += 1
        return {
            "total_events":    len(self.emitted_events),
            "event_types":     types,
            "staff_events":    staff_count,
            "unique_visitors": len({e["visitor_id"] for e in self.emitted_events}),
            **self.stats,
        }