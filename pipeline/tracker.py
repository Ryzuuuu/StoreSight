# PROMPT: Create a Re-ID gallery that maintains visitor appearance features across
# sessions for re-entry detection and cross-camera deduplication. Use HSV colour
# histogram as the appearance feature, cosine similarity for matching, with a
# configurable time window for re-entry detection.
#
# CHANGES MADE: Replaced unused class structure with a gallery actually imported
# and called by detect.py. Switched from ReIDTracker (never wired) to ReIDGallery
# (directly used). Added new_visitor() helper. Kept histogram approach (no GPU needed).

import cv2
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


class ReIDGallery:
    """
    Appearance-based Re-ID using normalised HSV colour histogram.
    Cosine similarity threshold 0.85 identifies re-entries within 30-min window.
    """
    SIM_THRESHOLD     = 0.85
    REENTRY_WIN_MIN   = 30

    def __init__(self):
        self._gallery: Dict[str, dict] = {}   # vid → {feats, exit_time}

    # ── Feature extraction ────────────────────────────────────────────────────
    def extract(self, frame: np.ndarray,
                bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 20:
            return np.zeros(192, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_h  = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        s_h  = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        v_h  = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        feat = np.concatenate([h_h, s_h, v_h]).astype(np.float32)
        nrm  = np.linalg.norm(feat)
        return feat / nrm if nrm > 1e-6 else feat

    # ── Matching ──────────────────────────────────────────────────────────────
    def match(self, feats: np.ndarray,
              now: datetime) -> Tuple[Optional[str], float]:
        """Return (visitor_id, similarity) if a re-entry match is found."""
        cutoff   = now - timedelta(minutes=self.REENTRY_WIN_MIN)
        best_vid, best_sim = None, 0.0
        for vid, entry in self._gallery.items():
            if entry["exit_time"] < cutoff:
                continue
            sim = float(np.dot(feats, entry["feats"]))
            if sim > best_sim:
                best_sim, best_vid = sim, vid
        if best_sim >= self.SIM_THRESHOLD:
            return best_vid, best_sim
        return None, 0.0

    # ── Session lifecycle ─────────────────────────────────────────────────────
    def close_session(self, visitor_id: str,
                      feats: np.ndarray, exit_time: datetime):
        """Archive a visitor's appearance features when they EXIT."""
        self._gallery[visitor_id] = {
            "feats":     feats,
            "exit_time": exit_time,
        }

    @staticmethod
    def new_visitor() -> str:
        return f"VIS_{uuid.uuid4().hex[:6].upper()}"