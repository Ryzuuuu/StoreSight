# Architectural Choices

Three key decisions, each with options considered, what AI suggested, and what was chosen and why.

---

## Decision 1 — Detection Model Selection

### Options considered
| Model | Pros | Cons |
|---|---|---|
| YOLOv8n | Fast, runs on CPU, single pip install, ByteTrack built-in | Smaller model, slightly lower mAP than YOLOv8m |
| YOLOv8m | Higher accuracy | 3× slower, needs GPU for real-time |
| RT-DETR | Transformer-based, strong on occlusion | Much slower, complex setup |
| MediaPipe | Very fast on CPU | Weaker on partial occlusion, no ByteTrack integration |

### What AI suggested
Claude suggested starting with YOLOv8n for speed during development and switching to YOLOv8m for final submission. It noted that for 20-minute retail clips at 15fps, YOLOv8m with GPU acceleration would produce meaningfully better detections in crowded billing scenes.

### What was chosen and why
**YOLOv8n with ByteTrack.** The `--model` flag in run_detection.py is configurable, so YOLOv8m can be substituted without code changes. YOLOv8n was chosen as the default because:
1. The challenge clips are 1080p/15fps retail CCTV — not fast action. Person detections at this resolution are well within YOLOv8n's capability.
2. The challenge runs on candidate hardware with no guaranteed GPU. YOLOv8n completes a 20-minute clip in ~20 minutes on CPU; YOLOv8m would take ~60 minutes.
3. ByteTrack (ReID-free tracker built into ultralytics) handles the multi-object tracking requirement with zero additional setup.

The AI suggestion to use YOLOv8m for the final run was acknowledged but not followed as default, to ensure the submission works on any hardware without timeouts.

**On VLMs for zone classification**: Claude suggested using a VLM (GPT-4V or Claude Vision) to classify which zone a person is standing in by describing the frame. This was evaluated and rejected. Zone classification via bounding box foot-point against store_layout.json rectangles is deterministic, zero-latency, and costs nothing at inference time. A VLM call per frame would add seconds of latency and API cost. The structured zone definitions in store_layout.json are precise enough to make rule-based classification accurate.

---

## Decision 2 — Event Schema Design

### Options considered
1. **Flat schema** — all fields at top level, no metadata nesting
2. **Spec schema** — exact structure from the problem statement with `metadata` sub-object
3. **Typed schema** — Pydantic EventType enum, strict typing throughout

### What AI suggested
Claude suggested adding a `session_id` field separate from `visitor_id` to distinguish between multiple visits by the same person. It also suggested making `zone_id` required (not nullable) and using a sentinel value like `"THRESHOLD"` for entry/exit events.

### What was chosen and why
**Exact spec schema with no additions.** The problem statement defines the schema precisely and states it will be validated by an automated harness. Adding undocumented fields risks breaking the test suite. The AI's `session_id` suggestion was rejected — `visitor_id` + `session_seq` already carries session identity. The `zone_id: null` for ENTRY/EXIT is explicit in the spec and was kept exactly as specified.

The `metadata` sub-object (`queue_depth`, `sku_zone`, `session_seq`) was kept strictly as defined. `session_seq` starts at 1 and increments before each emit, making the ordinal position unambiguous.

**On timestamp derivation**: The spec says "ISO-8601 UTC — derived from clip + frame offset". The implementation uses `clip_start_utc + (frame_number / fps)` as a `timedelta`. The `clip_start_utc` values are stored per camera in store_layout.json. The AI suggested reading timestamps from video EXIF metadata, but the retail CCTV clips did not have reliable embedded timestamps, making the config-driven approach more robust.

---

## Decision 3 — API Architecture and Storage

### Options considered
| Approach | Pros | Cons |
|---|---|---|
| In-memory Python list | Zero setup, passes acceptance gate, simple | Lost on restart, not scalable |
| SQLite | Persistent, SQL queries, zero infrastructure | Needs schema migration, slower for writes |
| PostgreSQL + Redis | Production-grade, fast reads/writes | Requires Docker services, complex setup |

### What AI suggested
Claude strongly recommended SQLite as the minimum for persistence, arguing that an in-memory store would lose all data on API restart and would fail any "restart and re-query" test scenario. It provided a full SQLAlchemy schema with indexes on `store_id`, `event_type`, and `timestamp`.

### What was chosen and why
**In-memory store for the challenge submission.** The reasoning:

1. The acceptance gate tests `POST /events/ingest` and `GET /stores/STORE_BLR_002/metrics` in a single session — no restart between ingest and query. The in-memory store passes this.
2. SQLite adds a `requirements.txt` entry (sqlalchemy), a migration step, and file permission concerns inside Docker. Each is a potential failure point that could cause an acceptance gate rejection.
3. The challenge explicitly says "SQLite is fine" in the FAQ — the evaluators are not expecting a production database.

The AI's recommendation was noted and partially followed: the storage abstraction (`EVENTS` list, `SEEN_IDS` set) is trivially replaceable with a database-backed implementation. A production upgrade path would be: `EVENTS → SQLite table → PostgreSQL` with no API endpoint changes required.

**On idempotency**: The AI correctly identified that idempotency must be enforced by `event_id`, not content hash. Two valid events with identical field values but different IDs should both be accepted. This was implemented exactly as suggested and verified by tests in `test_metrics.py::TestIngestIdempotency`.