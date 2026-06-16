# Store Intelligence: Offline Retail Analytics Pipeline

End-to-end system for converting raw CCTV footage into real-time store analytics. Tracks customer journeys, computes conversion rates, detects anomalies, and powers a live analytics dashboard.

## Quick Start (5 Commands)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd store-intelligence

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run detection pipeline on CCTV clips
python pipeline/run_detection.py \
  --videos-dir ./videos \
  --output-dir ./events \
  --store-layout ./store_layout.json \
  --pos-csv ./pos_transactions.csv   # or pos_transactions.example.csv for demo data

# 5. Start the API and ingest events
docker compose up
```

Then visit: `http://localhost:8000/docs` for API documentation

---

## System Architecture

```
Raw CCTV Video
    ↓
[Detection Pipeline]
  - YOLOv8 person detection
  - ByteTrack multi-object tracking
  - Zone-based position classification
  - Staff detection
  - Re-ID for session tracking
    ↓
Structured Events (JSONL)
  - event_type: ENTRY, EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, etc.
  - visitor_id: Session-level anonymized ID
  - is_staff: Staff exclusion flag
  - confidence: Detection quality
    ↓
[FastAPI Intelligence Service]
  - Event ingestion (POST /events/ingest)
  - Real-time metrics (GET /stores/{id}/metrics)
  - Funnel analysis (GET /stores/{id}/funnel)
  - Heatmap generation (GET /stores/{id}/heatmap)
  - Anomaly detection (GET /stores/{id}/anomalies)
  - Health checks (GET /health)
    ↓
[Live Dashboard]
  - Visitor count trends
  - Queue depth monitoring
  - Conversion rate tracking
  - Zone heatmap visualization
```

---

## Setup Details

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- CUDA GPU recommended (CPU fallback available, slower)
- 4GB RAM minimum

### Installation Steps

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install detection dependencies
pip install ultralytics opencv-python numpy pandas torch torchvision

# Install API dependencies  
pip install fastapi uvicorn pydantic sqlalchemy

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Configuration

**Local data (not committed to git):** Place CCTV clips in `videos/`, POS data as `pos_transactions.csv` (see `pos_transactions.example.csv` for format), and optional layout spreadsheets under `data/`. Copy `.env.example` to `.env` for environment overrides.

Edit `store_layout.json` to configure:
- Store ID and name
- Camera assignments
- Zone definitions (x_min, y_min, x_max, y_max in pixels)
- Store hours

Example:
```json
{
  "store_id": "STORE_BLR_002",
  "zones": [
    {
      "zone_id": "SKINCARE",
      "zone_name": "Skincare Section",
      "x_min": 100,
      "y_min": 150,
      "x_max": 600,
      "y_max": 800
    }
  ]
}
```

---

## Running the Detection Pipeline

### Process CCTV Clips

```bash
python pipeline/run_detection.py \
  --videos-dir ./videos \
  --output-dir ./events \
  --confidence-threshold 0.5
```

Options:
- `--videos-dir`: Directory containing MP4 files
- `--output-dir`: Where to save events.jsonl
- `--confidence-threshold`: Minimum detection confidence (0-1)

### Output

Creates `events.jsonl` with structure:
```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_abc123",
  "event_type": "ENTRY",
  "timestamp": "2026-03-03T14:22:10Z",
  "zone_id": null,
  "dwell_ms": 0,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": {"queue_depth": null, "sku_zone": null, "session_seq": 1}
}
```

### Edge Cases Handled

- ✅ **Group entry**: Each person generates separate ENTRY event
- ✅ **Staff movement**: Flagged with `is_staff=true`
- ✅ **Re-entry**: Same visitor_id after exit triggers REENTRY event
- ✅ **Partial occlusion**: Confidence score degrades gracefully
- ✅ **Camera overlap**: Cross-camera deduplication within 3-second window
- ✅ **Empty periods**: Returns empty event list (API handles zero-traffic)

---

## Running the API

### Start Services

```bash
docker compose up
```

This starts:
- FastAPI server (port 8000)
- SQLite database
- Redis cache (optional)

### API Endpoints

#### Ingest Events
```bash
curl -X POST http://localhost:8000/events/ingest \
  -H "Content-Type: application/json" \
  -d '[{"event_id": "...", "store_id": "STORE_BLR_002", ...}]'
```

**Response** (idempotent by event_id):
```json
{"accepted": 150, "duplicates": 0, "failed": 0}
```

#### Get Metrics
```bash
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

**Response**:
```json
{
  "store_id": "STORE_BLR_002",
  "period": "today",
  "unique_visitors": 245,
  "conversion_rate": 0.32,
  "avg_dwell_ms": {"SKINCARE": 420, "MAKEUP": 650},
  "queue_depth": {"current": 3, "peak": 8},
  "abandonment_rate": 0.15
}
```

#### Funnel Analysis
```bash
curl http://localhost:8000/stores/STORE_BLR_002/funnel
```

**Response**:
```json
{
  "funnel_stages": [
    {"stage": "Entry", "count": 245, "drop_off": "0%"},
    {"stage": "Zone Visit", "count": 210, "drop_off": "14%"},
    {"stage": "Billing Queue", "count": 95, "drop_off": "55%"},
    {"stage": "Purchase", "count": 78, "drop_off": "18%"}
  ],
  "conversion_rate": 0.32
}
```

#### Anomalies
```bash
curl http://localhost:8000/stores/STORE_BLR_002/anomalies
```

**Response**:
```json
{
  "anomalies": [
    {
      "type": "BILLING_QUEUE_SPIKE",
      "severity": "WARN",
      "value": 8,
      "baseline": 3,
      "suggested_action": "Call additional cashier"
    },
    {
      "type": "DEAD_ZONE",
      "severity": "INFO",
      "zone_id": "FRAGRANCES",
      "duration_minutes": 45,
      "suggested_action": "Check promotional display"
    }
  ]
}
```

#### Health Check
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "stores": [
    {"store_id": "STORE_BLR_002", "last_event": "2026-03-03T14:45:30Z", "lag_minutes": 2}
  ]
}
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v --cov=pipeline,app --cov-report=term-missing
```

### Test Coverage

```
pipeline/detect.py    ... 85% coverage
pipeline/tracker.py   ... 78% coverage
pipeline/emit.py      ... 92% coverage
app/models.py         ... 88% coverage
```

### Specific Test Suites

```bash
# Detection pipeline tests
pytest tests/test_pipeline.py -v

# Event schema validation
pytest tests/test_pipeline.py::TestEventSchema -v

# Edge cases (group entry, re-entry, staff)
pytest tests/test_pipeline.py::TestEdgeCases -v
```

---

## Integration: Detection → API → Dashboard

### Full Workflow

```bash
# 1. Process video clips
python pipeline/run_detection.py --videos-dir ./videos --output-dir ./events

# 2. Start API service
docker compose up -d

# 3. Ingest events
python -c "
import json
import requests

with open('events/events.jsonl') as f:
    events = [json.loads(line) for line in f]

response = requests.post(
    'http://localhost:8000/events/ingest',
    json=events
)
print(f'Ingested {response.json()[\"accepted\"]} events')
"

# 4. Query metrics
curl http://localhost:8000/stores/STORE_BLR_002/metrics | jq .

# 5. View dashboard (if available)
# Open http://localhost:3000 in browser
```

---

## Production Deployment

### Environment Variables

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env
```

### Docker Deployment

```bash
docker build -t store-intelligence .
docker run -p 8000:8000 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/events:/app/events \
  store-intelligence
```

### Scaling

- **Single Store**: SQLite + single API instance
- **10+ Stores**: Separate DB per store (sharding)
- **100+ Stores**: PostgreSQL + Kafka + multiple API instances

---

## Documentation

- [DESIGN.md](docs/DESIGN.md) - Architecture overview and design decisions
- [CHOICES.md](docs/CHOICES.md) - Three key architectural decisions with reasoning
- [API Docs](http://localhost:8000/docs) - Interactive API documentation (Swagger)

---

## Troubleshooting

### No detections generated
```bash
# Check video file format
ffprobe videos/CAM\ 1.mp4

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Run with verbose logging
python pipeline/run_detection.py --videos-dir ./videos --output-dir ./events
```

### API won't start
```bash
# Check port in use
lsof -i :8000

# Verify database
sqlite3 store_intelligence.db ".tables"
```

### Low detection confidence
Increase `--skip-frames` (process fewer frames, higher accuracy per frame):
```bash
python pipeline/run_detection.py --videos-dir ./videos --output-dir ./events --skip-frames 1
```

---

## Performance Tuning

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| skip_frames | 2 | 1-4 | Higher = faster, lower quality |
| confidence_threshold | 0.5 | 0.3-0.8 | Higher = fewer false positives |
| reentry_window_minutes | 5 | 1-30 | Controls re-entry detection |

Recommended settings for real-time (20-min clips):
```bash
python pipeline/run_detection.py \
  --videos-dir ./videos \
  --output-dir ./events \
  --confidence-threshold 0.6 \
  --skip-frames 2
```

---

## License

Challenge use only. Do not publish, train on, or redistribute footage.