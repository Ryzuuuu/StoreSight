"""
Store Intelligence API — minimal FastAPI entrypoint.
Acceptance gate requirements:
  POST /events/ingest          → 200 with accepted/duplicate/failed counts
  GET  /stores/{store_id}/metrics → 200 with visitor + conversion metrics
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="Store Intelligence API", version="1.0.0")

# ── In-memory store ───────────────────────────────────────────────────────────
EVENTS:        List[dict] = []
SEEN_IDS:      set        = set()
STORE_REGISTRY: set       = set()


# ── Middleware: structured request logging ────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id   = str(uuid.uuid4())[:8]
    start      = time.time()
    response   = await call_next(request)
    latency_ms = int((time.time() - start) * 1000)
    logger.info(
        f"trace_id={trace_id} endpoint={request.url.path} "
        f"method={request.method} status={response.status_code} "
        f"latency_ms={latency_ms}"
    )
    return response


# ── Models ────────────────────────────────────────────────────────────────────
class EventMetadata(BaseModel):
    queue_depth:  Optional[int]  = None
    sku_zone:     Optional[str]  = None
    session_seq:  int            = 1

class VisitorEvent(BaseModel):
    event_id:   str
    store_id:   str
    camera_id:  str
    visitor_id: str
    event_type: str
    timestamp:  str
    zone_id:    Optional[str]    = None
    dwell_ms:   int              = 0
    is_staff:   bool             = False
    confidence: float
    metadata:   EventMetadata    = Field(default_factory=EventMetadata)


# ── Ingest ────────────────────────────────────────────────────────────────────
@app.post("/events/ingest")
async def ingest(events: List[VisitorEvent]):
    accepted = duplicate = failed = 0
    errors: List[dict] = []

    for ev in events:
        try:
            if ev.event_id in SEEN_IDS:
                duplicate += 1
                continue
            SEEN_IDS.add(ev.event_id)
            d = ev.dict()
            EVENTS.append(d)
            STORE_REGISTRY.add(ev.store_id)
            accepted += 1
        except Exception as e:
            failed += 1
            errors.append({"event_id": getattr(ev, "event_id", "?"), "error": str(e)})

    return {"accepted": accepted, "duplicates": duplicate, "failed": failed,
            "errors": errors}


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/stores/{store_id}/metrics")
async def metrics(store_id: str):
    store_events = [e for e in EVENTS
                    if e["store_id"] == store_id and not e["is_staff"]]

    unique_visitors = len({e["visitor_id"]
                           for e in store_events if e["event_type"] == "ENTRY"})
    entries   = sum(1 for e in store_events if e["event_type"] == "ENTRY")
    purchases = len({e["visitor_id"] for e in store_events if e["event_type"] == "BILLING_QUEUE_JOIN"})
    conversion_rate = round(purchases / unique_visitors, 4) if unique_visitors else 0.0

    # Avg dwell per zone
    zone_dwell: Dict[str, List[int]] = {}
    for e in store_events:
        if e["event_type"] in ("ZONE_EXIT", "ZONE_DWELL") and e.get("zone_id"):
            zone_dwell.setdefault(e["zone_id"], []).append(e["dwell_ms"])
    avg_dwell = {z: int(sum(v)/len(v)) for z, v in zone_dwell.items()}

    # Current billing queue depth
    billing_joins = sum(1 for e in store_events if e["event_type"] == "BILLING_QUEUE_JOIN")
    abandons      = sum(1 for e in store_events if e["event_type"] == "BILLING_QUEUE_ABANDON")
    abandon_rate  = round(abandons / billing_joins, 4) if billing_joins else 0.0

    return {
        "store_id":        store_id,
        "period":          "today",
        "unique_visitors": unique_visitors,
        "total_entries":   entries,
        "conversion_rate": conversion_rate,
        "avg_dwell_ms":    avg_dwell,
        "queue_depth":     {"current": max(0, billing_joins - abandons)},
        "abandonment_rate": abandon_rate,
        "data_from_events": len(store_events),
    }


# ── Funnel ────────────────────────────────────────────────────────────────────
@app.get("/stores/{store_id}/funnel")
async def funnel(store_id: str):
    evts = [e for e in EVENTS if e["store_id"] == store_id and not e["is_staff"]]
    sessions = {e["visitor_id"] for e in evts if e["event_type"] == "ENTRY"}
    zone_visitors = {e["visitor_id"] for e in evts if e["event_type"] == "ZONE_ENTER"}
    billing_visitors = {e["visitor_id"] for e in evts
                        if e["event_type"] in ("BILLING_QUEUE_JOIN", "ZONE_ENTER")
                        and e.get("zone_id") == "BILLING"}
    # Purchases: visitors who had billing join but no abandon
    abandoned = {e["visitor_id"] for e in evts if e["event_type"] == "BILLING_QUEUE_ABANDON"}
    purchasers = billing_visitors - abandoned

    n_entry   = len(sessions)
    n_zone    = len(zone_visitors & sessions)
    n_billing = len(billing_visitors & sessions)
    n_purchase= len(purchasers & sessions)

    def pct(a, b): return f"{round((1 - a/b)*100)}%" if b else "0%"

    return {
        "funnel_stages": [
            {"stage": "Entry",        "count": n_entry,    "drop_off": "0%"},
            {"stage": "Zone Visit",   "count": n_zone,     "drop_off": pct(n_zone,    n_entry)},
            {"stage": "Billing Queue","count": n_billing,  "drop_off": pct(n_billing, n_zone)},
            {"stage": "Purchase",     "count": n_purchase, "drop_off": pct(n_purchase,n_billing)},
        ],
        "conversion_rate": round(n_purchase / n_entry, 4) if n_entry else 0.0,
    }


# ── Heatmap ───────────────────────────────────────────────────────────────────
@app.get("/stores/{store_id}/heatmap")
async def heatmap(store_id: str):
    evts = [e for e in EVENTS if e["store_id"] == store_id and not e["is_staff"]]
    zone_visits: Dict[str, int]       = {}
    zone_dwell:  Dict[str, List[int]] = {}
    for e in evts:
        z = e.get("zone_id")
        if not z:
            continue
        if e["event_type"] == "ZONE_ENTER":
            zone_visits[z] = zone_visits.get(z, 0) + 1
        if e["event_type"] in ("ZONE_EXIT", "ZONE_DWELL") and e["dwell_ms"]:
            zone_dwell.setdefault(z, []).append(e["dwell_ms"])

    max_v = max(zone_visits.values(), default=1)
    sessions = len({e["visitor_id"] for e in evts if e["event_type"] == "ENTRY"})
    result = {}
    for z, visits in zone_visits.items():
        avg = int(sum(zone_dwell.get(z, [0])) / max(len(zone_dwell.get(z, [1])), 1))
        result[z] = {
            "visit_frequency_norm": round(visits / max_v * 100),
            "avg_dwell_ms":         avg,
            "data_confidence":      sessions >= 20,
        }
    return {"store_id": store_id, "zones": result}


# ── Anomalies ─────────────────────────────────────────────────────────────────
@app.get("/stores/{store_id}/anomalies")
async def anomalies(store_id: str):
    evts = [e for e in EVENTS if e["store_id"] == store_id and not e["is_staff"]]
    anomaly_list = []

    # Queue spike: current depth > 3
    billing_joins = sum(1 for e in evts if e["event_type"] == "BILLING_QUEUE_JOIN")
    abandons      = sum(1 for e in evts if e["event_type"] == "BILLING_QUEUE_ABANDON")
    current_q     = max(0, billing_joins - abandons)
    if current_q > 3:
        anomaly_list.append({
            "type": "BILLING_QUEUE_SPIKE", "severity": "WARN",
            "value": current_q, "baseline": 2,
            "suggested_action": "Call additional cashier to billing counter",
        })

    # Dead zone: any zone with no visits
    zone_visits = {e.get("zone_id") for e in evts if e["event_type"] == "ZONE_ENTER"}
    all_zones   = {"SKINCARE_BACK","MAKEUP","BILLING","ACCESSORIES","FRAGRANCE"}
    for z in (all_zones - zone_visits):
        anomaly_list.append({
            "type": "DEAD_ZONE", "severity": "INFO",
            "zone_id": z, "duration_minutes": 30,
            "suggested_action": f"Check display and signage for {z}",
        })

    # Conversion drop
    entries   = len({e["visitor_id"] for e in evts if e["event_type"] == "ENTRY"})
    purchases = len({e["visitor_id"] for e in evts
                     if e["event_type"] == "BILLING_QUEUE_JOIN"})
    if entries > 10 and purchases / max(entries, 1) < 0.10:
        anomaly_list.append({
            "type": "CONVERSION_DROP", "severity": "CRITICAL",
            "value": round(purchases/entries, 3), "baseline": 0.25,
            "suggested_action": "Review floor staff engagement and promotions",
        })

    return {"store_id": store_id, "anomalies": anomaly_list}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    now = datetime.now(timezone.utc)
    store_health = []
    for sid in STORE_REGISTRY:
        store_evts = [e for e in EVENTS if e["store_id"] == sid]
        if store_evts:
            last_ts_str = max(e["timestamp"] for e in store_evts)
            last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
            lag_min = int((now - last_ts).total_seconds() / 60)
            status  = "STALE_FEED" if lag_min > 10 else "ok"
        else:
            last_ts_str, lag_min, status = None, None, "no_events"
        store_health.append({
            "store_id": sid, "last_event": last_ts_str,
            "lag_minutes": lag_min, "status": status,
        })
    return {
        "status":      "healthy",
        "total_events": len(EVENTS),
        "stores":       store_health,
        "timestamp":    now.isoformat(),
    }