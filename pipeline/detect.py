# PROMPT: Build a real CCTV person detection pipeline using YOLOv8 + ByteTrack that emits
# all 8 event types: ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN,
# BILLING_QUEUE_ABANDON, REENTRY. Requirements: staff detection via HSV uniform analysis,
# Re-ID for session continuity and re-entry detection, POS correlation for abandonment,
# virtual entry line crossing for directional entry/exit, per-person dwell tracking,
# correct timestamp from clip_start + frame offset, confidence always clamped 0-1.
#
# CHANGES MADE: Replaced mock detection with real YOLOv8+ByteTrack, added all missing
# event types (EXIT, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY), fixed timestamp
# from hardcoded date to clip_start+offset, fixed confidence overflow bug, wired
# ReIDGallery from tracker.py, fixed session_seq counter, fixed ZONE_DWELL to track
# per-person zone entry time not total clip time.

import json, cv2, uuid, logging, csv
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from ultralytics import YOLO
from tracker import ReIDGallery

logger = logging.getLogger(__name__)

DWELL_INTERVAL_MS = 30_000
MIN_DWELL_MS      = 2_000
ABANDON_WINDOW_S  = 300


class PersonTracker:

    def __init__(self, store_layout_path: str,
                 pos_csv_path: Optional[str] = None,
                 model_name: str = "yolov8n.pt",
                 confidence_threshold: float = 0.4):

        with open(store_layout_path) as f:
            self.layout = json.load(f)

        self.store_id   = self.layout["store_id"]
        self.zones      = {z["zone_id"]: z for z in self.layout.get("zones", [])}
        self.cameras    = {c["camera_id"]: c for c in self.layout.get("cameras", [])}
        self.conf_thr   = confidence_threshold
        self.pos_txns   = self._load_pos(pos_csv_path) if pos_csv_path else []
        self.reid       = ReIDGallery()
        self.events_buffer: List[dict] = []

        logger.info(f"Loading {model_name} ...")
        self.model = YOLO(model_name)

    # ── POS ──────────────────────────────────────────────────────────────────
    def _load_pos(self, path: str) -> List[dict]:
        txns = []
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        raw = f"{row['order_date']} {row['order_time']}"
                        dt  = datetime.strptime(raw, "%d-%m-%Y %H:%M:%S")
                        dt_utc = (dt - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
                        txns.append({"ts": dt_utc, "amt": float(row.get("total_amount") or 0)})
                    except Exception:
                        pass
            logger.info(f"Loaded {len(txns)} POS transactions")
        except Exception as e:
            logger.warning(f"POS load failed: {e}")
        return txns

    # ── Zone ─────────────────────────────────────────────────────────────────
    def _get_zone(self, foot_x: float, foot_y: float, camera_id: str) -> Optional[str]:
        candidates = [z for z in self.zones.values()
                      if z.get("camera_id") == camera_id
                      and z.get("type") != "entry_point"]
        # Higher-priority types first
        candidates.sort(key=lambda z: 0 if z["type"] in ("checkout", "product_zone") else 1)
        for z in candidates:
            if z["x_min"] <= foot_x <= z["x_max"] and z["y_min"] <= foot_y <= z["y_max"]:
                return z["zone_id"]
        return None

    # ── Staff ─────────────────────────────────────────────────────────────────
    # Detect staff via uniform colour consistency on the torso crop.
    # We check three candidate uniform colours common in Indian retail:
    #   1. Black / very dark (V < 70)
    #   2. Pink / fuchsia (Purplle brand — H 140-175, S > 100)
    #   3. Dark navy / royal blue (H 100-130, S > 80, V 50-160)
    # If ANY colour dominates ≥ 30 % of the torso, flag as staff.
    def _is_staff(self, frame: np.ndarray, bbox: Tuple) -> Tuple[bool, float]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        yt = max(0, y1 + (y2-y1)//3)
        yb = min(h, y1 + 2*(y2-y1)//3)
        xt = max(0, x1+4); xb = min(w, x2-4)
        if xb <= xt or yb <= yt:
            return False, 0.0
        torso = frame[yt:yb, xt:xb]
        if torso.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total = float(hsv[:,:,0].size)

        # 1. Black / very dark uniform (tightened: V<60)
        m_black = cv2.inRange(hsv, np.array([0,   0,  0]), np.array([180, 120, 60]))
        # 2. Pink / fuchsia (Purplle brand — narrow hue range)
        m_pink  = cv2.inRange(hsv, np.array([145, 100, 90]), np.array([172, 255, 255]))
        # 3. Dark navy / blue (tightened)
        m_blue  = cv2.inRange(hsv, np.array([105, 90, 45]), np.array([128, 255, 150]))

        ratios = [np.count_nonzero(m) / total for m in (m_black, m_pink, m_blue)]
        best   = max(ratios)
        # Raised threshold 30% → 45% to reduce false-positives on dark-clothed customers
        score  = float(np.clip(best / 0.45, 0.0, 1.0))
        return score >= 1.0, round(best, 3)   # triggers at 45% coverage

    # ── Entry line crossing ───────────────────────────────────────────────────
    def _check_crossing(self, cam_cfg: dict,
                        prev_cx: float, curr_cx: float,
                        prev_cy: float, curr_cy: float,
                        tid: int, debounce_state: dict, frame_num: int) -> Optional[str]:
        # Support vertical line (entry_line_x) OR horizontal line (entry_line_y)
        line_x = cam_cfg.get("entry_line_x")
        line_y = cam_cfg.get("entry_line_y")

        if line_x is not None:
            # Vertical line: inside = left (x < line_x)
            was_inside = prev_cx < line_x
            now_inside = curr_cx < line_x
            coord_dist = abs(curr_cx - line_x)
            hysteresis = cam_cfg.get("hysteresis_px", 30)
        elif line_y is not None:
            # Horizontal line: inside = above line (y < line_y, since y increases downward)
            was_inside = prev_cy < line_y
            now_inside = curr_cy < line_y
            coord_dist = abs(curr_cy - line_y)
            hysteresis = cam_cfg.get("hysteresis_py", cam_cfg.get("hysteresis_px", 30))
        else:
            return None

        if was_inside == now_inside:
            return None
        if coord_dist < hysteresis:
            return None

        direction = "ENTRY" if not was_inside else "EXIT"
        debounce  = cam_cfg.get("debounce_frames", 30)
        key       = f"{tid}_{direction}"
        if frame_num - debounce_state.get(key, -9999) < debounce:
            return None
        debounce_state[key] = frame_num
        return direction

    # ── Timestamp ────────────────────────────────────────────────────────────
    @staticmethod
    def _ts(frame_num: int, fps: float, clip_start: datetime) -> str:
        dt = clip_start + timedelta(seconds=frame_num / fps)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Emit ─────────────────────────────────────────────────────────────────
    def _emit(self, event_type: str, visitor_id: str, camera_id: str,
              zone_id: Optional[str], frame_num: int, fps: float,
              clip_start: datetime, confidence: float, is_staff: bool,
              dwell_ms: int = 0, queue_depth: Optional[int] = None,
              seq: int = 1) -> dict:
        event = {
            "event_id":   str(uuid.uuid4()),
            "store_id":   self.store_id,
            "camera_id":  camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp":  self._ts(frame_num, fps, clip_start),
            "zone_id":    zone_id,
            "dwell_ms":   dwell_ms,
            "is_staff":   is_staff,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
            "metadata": {
                "queue_depth": queue_depth,
                "sku_zone":    self.zones.get(zone_id, {}).get("sku_zone") if zone_id else None,
                "session_seq": seq,
            },
        }
        self.events_buffer.append(event)
        return event

    # ── Main ─────────────────────────────────────────────────────────────────
    def process_video_clip(self, video_path: str, camera_id: str,
                           clip_start: datetime, skip_frames: int = 2) -> List[dict]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open {video_path}")
            return []

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cam   = self.cameras.get(camera_id, {})
        is_entry   = cam.get("is_entry_camera", False)
        is_backroom= cam.get("is_backroom", False)
        logger.info(f"[{camera_id}] {total} frames @ {fps:.1f}fps  entry={is_entry}")

        # Per-track state
        track_vid:          Dict[int, str]           = {}
        track_feats:        Dict[int, np.ndarray]    = {}
        track_staff:        Dict[int, bool]          = {}
        track_prev_cx:      Dict[int, float]         = {}
        track_prev_cy:      Dict[int, float]         = {}
        track_zone:         Dict[int, Optional[str]] = {}
        track_zone_frame:   Dict[int, int]           = {}
        track_dwell_n:      Dict[int, int]           = {}
        track_entered:      set                      = set()
        track_exited:       set                      = set()
        track_seq:          Dict[str, int]           = defaultdict(int)
        billing_entry_time: Dict[str, datetime]      = {}
        debounce:           Dict[str, int]           = {}

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            if frame_num % skip_frames != 0:
                continue

            now_dt = clip_start + timedelta(seconds=frame_num / fps)

            results = self.model.track(
                frame, conf=self.conf_thr, iou=0.45, classes=[0],
                tracker="bytetrack.yaml", persist=True, verbose=False, imgsz=640,
            )

            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            current_ids = set(ids.tolist())

            # Lost tracks → implicit zone exit
            for tid in list(track_vid.keys()):
                if tid not in current_ids:
                    self._handle_lost(tid, frame_num, fps, clip_start, camera_id,
                                      track_vid, track_staff, track_zone,
                                      track_zone_frame, track_seq,
                                      billing_entry_time, now_dt, is_entry, is_backroom,
                                      track_entered, track_exited, track_feats)
                    for d in (track_vid, track_feats, track_staff, track_prev_cx,
                              track_prev_cy, track_zone, track_zone_frame, track_dwell_n):
                        d.pop(tid, None)

            for bbox, tid, det_conf in zip(boxes, ids, confs):
                tid  = int(tid)
                x1, y1, x2, y2 = bbox
                cx   = (x1 + x2) / 2.0
                foot_x, foot_y = cx, float(y2)
                conf = float(np.clip(det_conf, 0.0, 1.0))

                # ── Assign visitor_id ────────────────────────────────────
                if tid not in track_vid:
                    feats = self.reid.extract(frame, bbox)
                    matched_vid, sim = self.reid.match(feats, now_dt)
                    if matched_vid:
                        vid        = matched_vid
                        is_reentry = True
                        conf       = float(np.clip(sim, 0.0, 1.0))
                    else:
                        vid        = self.reid.new_visitor()
                        is_reentry = False
                    track_vid[tid]   = vid
                    track_feats[tid] = feats
                    track_staff[tid] = False
                else:
                    vid        = track_vid[tid]
                    is_reentry = False
                    # Update appearance features periodically
                    if frame_num % 30 == 0:
                        track_feats[tid] = self.reid.extract(frame, bbox)

                # ── Staff detection every 15 frames ─────────────────────
                if frame_num % 15 == 1:
                    sf, _ = self._is_staff(frame, bbox)
                    track_staff[tid] = sf
                    if sf and vid.startswith("VIS_"):
                        new_vid = "STF_" + vid[4:]
                        track_vid[tid] = new_vid
                        vid = new_vid

                is_sf = track_staff.get(tid, False)

                # Backroom: staff only, skip customer events
                if is_backroom:
                    track_staff[tid] = True
                    track_prev_cx[tid] = cx
                    continue

                # ── Entry/Exit: appearance = ENTRY, disappearance = EXIT ───
                # Works for any camera angle (overhead, angled, side-on).
                # If entry_line_x or entry_line_y is set, uses line crossing instead.
                if is_entry:
                    cy = (y1 + y2) / 2.0
                    use_line = cam.get("entry_line_x") is not None or cam.get("entry_line_y") is not None

                    if use_line and tid in track_prev_cx:
                        direction = self._check_crossing(
                            cam, track_prev_cx[tid], cx,
                            track_prev_cy.get(tid, cy), cy,
                            tid, debounce, frame_num)
                        if direction == "ENTRY" and tid not in track_entered:
                            track_entered.add(tid)
                            track_exited.discard(tid)
                            etype = "REENTRY" if is_reentry else "ENTRY"
                            track_seq[vid] += 1
                            self._emit(etype, vid, camera_id, None,
                                       frame_num, fps, clip_start, conf, is_sf,
                                       seq=track_seq[vid])
                        elif direction == "EXIT" and tid in track_entered and tid not in track_exited:
                            track_exited.add(tid)
                            track_seq[vid] += 1
                            self._emit("EXIT", vid, camera_id, None,
                                       frame_num, fps, clip_start, conf, is_sf,
                                       seq=track_seq[vid])
                            self.reid.close_session(vid, track_feats.get(tid,
                                                    np.zeros(192)), now_dt)
                    elif not use_line:
                        # Appearance mode: new track = ENTRY
                        if tid not in track_entered:
                            track_entered.add(tid)
                            etype = "REENTRY" if is_reentry else "ENTRY"
                            track_seq[vid] += 1
                            self._emit(etype, vid, camera_id, None,
                                       frame_num, fps, clip_start, conf, is_sf,
                                       seq=track_seq[vid])
                    track_prev_cy[tid] = cy

                # ── Zone tracking (non-entry cameras) ────────────────────
                if not is_entry:
                    new_zone = self._get_zone(foot_x, foot_y, camera_id)
                    old_zone = track_zone.get(tid)

                    if new_zone != old_zone:
                        # Zone exit
                        if old_zone is not None:
                            dwell_ms = int((frame_num - track_zone_frame.get(tid, frame_num))
                                          / fps * 1000)
                            if dwell_ms >= MIN_DWELL_MS:
                                track_seq[vid] += 1
                                self._emit("ZONE_EXIT", vid, camera_id, old_zone,
                                           frame_num, fps, clip_start, conf, is_sf,
                                           dwell_ms=dwell_ms, seq=track_seq[vid])
                            if old_zone == "BILLING":
                                self._check_abandon(vid, billing_entry_time, now_dt,
                                                    frame_num, fps, clip_start,
                                                    camera_id, conf, is_sf, track_seq)

                        # Zone enter
                        if new_zone is not None:
                            track_zone_frame[tid] = frame_num
                            track_dwell_n[tid]    = 0
                            q_depth = sum(1 for z in track_zone.values() if z == "BILLING")

                            if new_zone == "BILLING":
                                billing_entry_time[vid] = now_dt
                                if q_depth > 0:
                                    track_seq[vid] += 1
                                    self._emit("BILLING_QUEUE_JOIN", vid, camera_id,
                                               "BILLING", frame_num, fps, clip_start,
                                               conf, is_sf, queue_depth=q_depth,
                                               seq=track_seq[vid])
                                else:
                                    track_seq[vid] += 1
                                    self._emit("ZONE_ENTER", vid, camera_id, new_zone,
                                               frame_num, fps, clip_start, conf, is_sf,
                                               seq=track_seq[vid])
                            else:
                                track_seq[vid] += 1
                                self._emit("ZONE_ENTER", vid, camera_id, new_zone,
                                           frame_num, fps, clip_start, conf, is_sf,
                                           seq=track_seq[vid])

                        track_zone[tid] = new_zone

                    # Dwell timer
                    elif new_zone is not None:
                        dwell_ms = int((frame_num - track_zone_frame.get(tid, frame_num))
                                      / fps * 1000)
                        interval_n = dwell_ms // DWELL_INTERVAL_MS
                        if interval_n > track_dwell_n.get(tid, 0) and interval_n > 0:
                            track_dwell_n[tid] = interval_n
                            track_seq[vid] += 1
                            self._emit("ZONE_DWELL", vid, camera_id, new_zone,
                                       frame_num, fps, clip_start, conf, is_sf,
                                       dwell_ms=dwell_ms, seq=track_seq[vid])

                track_prev_cx[tid] = cx

        cap.release()
        logger.info(f"[{camera_id}] finished. Events so far: {len(self.events_buffer)}")
        return self.events_buffer

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _handle_lost(self, tid, frame_num, fps, clip_start, camera_id,
                     track_vid, track_staff, track_zone, track_zone_frame,
                     track_seq, billing_entry_time, now_dt, is_entry, is_backroom,
                     track_entered=None, track_exited=None, track_feats=None):
        vid  = track_vid.get(tid)
        if not vid:
            return
        is_sf = track_staff.get(tid, False)

        # Entry camera: disappearance = EXIT (appearance mode)
        if is_entry and not is_backroom and not is_sf:
            entered = track_entered or set()
            exited  = track_exited  or set()
            if tid in entered and tid not in exited:
                exited.add(tid)
                track_seq[vid] += 1
                self._emit("EXIT", vid, camera_id, None,
                           frame_num, fps, clip_start, 0.60, is_sf,
                           seq=track_seq[vid])
                feats = (track_feats or {}).get(tid, np.zeros(192))
                self.reid.close_session(vid, feats, now_dt)

        if not is_entry and not is_backroom:
            old_zone = track_zone.get(tid)
            if old_zone:
                dwell_ms = int((frame_num - track_zone_frame.get(tid, frame_num))
                               / fps * 1000)
                if dwell_ms >= MIN_DWELL_MS:
                    track_seq[vid] += 1
                    self._emit("ZONE_EXIT", vid, camera_id, old_zone,
                               frame_num, fps, clip_start, 0.60, is_sf,
                               dwell_ms=dwell_ms, seq=track_seq[vid])
                if old_zone == "BILLING":
                    self._check_abandon(vid, billing_entry_time, now_dt,
                                        frame_num, fps, clip_start,
                                        camera_id, 0.60, is_sf, track_seq)

    def _check_abandon(self, vid, billing_entry_time, now_dt,
                       frame_num, fps, clip_start, camera_id, conf, is_sf, track_seq):
        entry_t = billing_entry_time.pop(vid, None)
        if entry_t is None:
            return
        window_end = now_dt + timedelta(seconds=ABANDON_WINDOW_S)
        purchased  = any(entry_t <= t["ts"] <= window_end for t in self.pos_txns)
        if not purchased:
            track_seq[vid] += 1
            self._emit("BILLING_QUEUE_ABANDON", vid, camera_id, "BILLING",
                       frame_num, fps, clip_start, conf, is_sf,
                       seq=track_seq[vid])

    def get_events(self) -> List[dict]:
        return self.events_buffer

    def save_events(self, output_path: str):
        import json as _json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for e in self.events_buffer:
                f.write(_json.dumps(e) + "\n")
        logger.info(f"Saved {len(self.events_buffer)} events → {output_path}")