# System Design — Store Intelligence Pipeline

## Architecture Overview

The system converts raw CCTV footage into live store analytics through a four-stage pipeline:

```
Raw CCTV Clips
      ↓
Detection Layer  (YOLOv8 + ByteTrack + ReIDGallery)
      ↓
events.jsonl     (structured behavioural events)
      ↓
Intelligence API (FastAPI — ingest, metrics, funnel, anomalies, health)
      ↓
Live Dashboard   (terminal rich / web UI)
```

Each stage is independently runnable. The detection pipeline is batch (offline); the API is stateful in-memory with a clean REST surface.

---

## Stage 1 — Detection Layer (`pipeline/`)

### Person detection and tracking
YOLOv8n runs on every second frame (configurable via `--skip-frames`) and returns bounding boxes for the `person` class. ByteTrack (built into ultralytics) assigns persistent track IDs across frames within a clip. Each unique track ID gets a `visitor_id` token on first appearance.

### Entry / Exit
CAM_3 is designated the entry camera (`is_entry_camera: true` in store_layout.json). When a new ByteTrack ID appears in this camera, an `ENTRY` event is emitted. When the track disappears, an `EXIT` event is emitted and the visitor's appearance features are archived in the Re-ID gallery. This appearance-based approach handles any camera mounting angle without requiring a calibrated virtual line.

### Zone classification
For floor and billing cameras (CAM_1, CAM_2, CAM_5), the foot point (bottom-centre of the bounding box) is tested against axis-aligned zone rectangles defined in store_layout.json. Higher-priority zone types (checkout, product_zone) are checked before floor zones to resolve overlap ambiguities.

### Staff detection
The torso region (middle vertical third of the bounding box) is extracted and converted to HSV. Three uniform colour profiles are tested: black/dark (V < 70), pink/fuchsia (Purplle brand H 140–175), and dark navy (H 100–130). If any colour covers ≥ 30 % of the torso pixels, the person is flagged `is_staff=true` and their visitor_id is prefixed `STF_`. Staff events are still emitted so operational data is preserved; the API excludes them from customer metrics.

### Re-ID and re-entry
`ReIDGallery` (tracker.py) stores a 192-dimensional normalised HSV histogram per exited visitor. When a new track appears, cosine similarity is computed against all gallery entries within a 30-minute window. Similarity ≥ 0.85 triggers a `REENTRY` event instead of `ENTRY` and preserves the original `visitor_id`.

### POS correlation
`pos_transactions.csv` is loaded and parsed at startup (IST → UTC). When a visitor leaves the BILLING zone, the pipeline checks whether any POS transaction occurred within the 5-minute window starting at their billing entry time. If none is found, `BILLING_QUEUE_ABANDON` is emitted.

### Dwell tracking
Each track maintains a zone entry frame. ZONE_DWELL is emitted once per 30-second interval of continuous dwell (`dwell_ms // 30000 > previously_emitted_count`), matching the spec exactly.

---

## Stage 2 — Event Schema (`pipeline/emit.py`, `app/models.py`)

Events follow the exact schema from the problem statement. Key design choices:
- `event_id`: UUID v4, generated at emission time — globally unique
- `timestamp`: `clip_start_utc + frame_number / fps` — derived from wall-clock metadata, not system time
- `confidence`: always clamped to [0, 1]; low-confidence events are emitted, never suppressed
- `session_seq`: per-visitor ordinal counter, incremented before each emit call

Deduplication in the emitter is by `event_id` (not content hash), so two valid events with identical content but different IDs are both accepted.

---

## Stage 3 — Intelligence API (`app/main.py`)

FastAPI with a single-file entrypoint. All state lives in an in-memory Python list (`EVENTS`) and set (`SEEN_IDS`). This trades persistence for simplicity — acceptable for a challenge submission; a production system would use PostgreSQL + Redis.

### Endpoints
| Endpoint | Notes |
|---|---|
| `POST /events/ingest` | Idempotent by `event_id`. Partial success — malformed events return structured errors, not 5xx. |
| `GET /stores/{id}/metrics` | Filters `is_staff=true`. `unique_visitors` counts only `ENTRY` events; `REENTRY` does not double-count. `conversion_rate` = unique billing visitors / unique entry visitors. |
| `GET /stores/{id}/funnel` | Session-based. Each stage counts unique `visitor_id`s. Drop-off is expressed as percentage relative to the previous stage. |
| `GET /stores/{id}/heatmap` | Zone visit frequency normalised 0–100. `data_confidence: false` when fewer than 20 sessions. |
| `GET /stores/{id}/anomalies` | Three anomaly types: `BILLING_QUEUE_SPIKE` (depth > 3), `DEAD_ZONE` (no visits in window), `CONVERSION_DROP` (rate < 10 % on > 10 sessions). All include `severity` and `suggested_action`. |
| `GET /health` | Per-store last event timestamp. `STALE_FEED` if lag > 10 minutes. |

### Structured logging
Every request logs `trace_id`, `endpoint`, `method`, `status_code`, and `latency_ms` via middleware.

---

## Stage 4 — Live Dashboard

The detection pipeline can feed events into the API in simulated real-time using `--skip-frames 1`. A separate `dashboard.py` (see README) polls `GET /stores/STORE_BLR_002/metrics` every 5 seconds and renders a live terminal display using the `rich` library.

---

## AI-Assisted Decisions

### 1. Appearance-mode entry/exit vs virtual line crossing
The original implementation used a vertical virtual line at x=870 for entry/exit detection. After observing 0 ENTRY events in initial runs, an LLM was used to diagnose the issue — it correctly identified that a line-crossing approach requires a camera mounted perpendicular to foot traffic. For an overhead or angled entry camera (which CAM_3 is), track appearance = ENTRY and track disappearance = EXIT is more reliable. This suggestion was adopted and confirmed to produce correct counts.

### 2. Staff detection colour range
The initial implementation detected only black uniforms (HSV V < 80). An LLM suggested expanding to cover Purplle's brand pink/fuchsia colour (HSV H 140–175) and dark navy. The coverage threshold was kept at 30 % rather than the LLM's suggested 20 % to reduce false positives on dark-clothed customers.

### 3. POS correlation window direction
The LLM initially suggested checking transactions in the 5-minute window *before* billing exit, matching the PDF's wording literally. After review, the implementation checks from billing *entry* time to 5 minutes after exit — a broader window that is more tolerant of slow billing processes. The LLM's stricter window was overridden.