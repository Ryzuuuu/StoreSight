#!/usr/bin/env python3
"""
Detection pipeline orchestrator.
Reads clip_start_utc from store_layout.json per camera.
Maps video filenames to camera_ids via store_layout cameras[].file field.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from detect import PersonTracker
from emit   import EventEmitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_clip_start(iso_str: str) -> datetime:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--videos-dir",           required=True)
    parser.add_argument("--output-dir",           required=True)
    parser.add_argument("--store-layout",         default="store_layout.json")
    parser.add_argument("--pos-csv",              default=None)
    parser.add_argument("--model",                default="yolov8n.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--skip-frames",          type=int,   default=2)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return 1

    with open(args.store_layout) as f:
        layout = json.load(f)

    # Build filename → (camera_id, clip_start_utc) lookup
    cam_map = {}
    for cam in layout.get("cameras", []):
        fname = cam.get("file", "")
        cam_map[fname] = {
            "camera_id":   cam["camera_id"],
            "clip_start":  parse_clip_start(cam.get("clip_start_utc",
                                             "2026-04-10T14:40:00Z")),
        }

    logger.info("Initialising detection pipeline...")
    tracker = PersonTracker(
        store_layout_path    = args.store_layout,
        pos_csv_path         = args.pos_csv,
        model_name           = args.model,
        confidence_threshold = args.confidence_threshold,
    )
    emitter = EventEmitter()

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        logger.error(f"No MP4 files found in {videos_dir}")
        return 1

    logger.info(f"Found {len(video_files)} video file(s)")

    for vf in video_files:
        info = cam_map.get(vf.name)
        if info is None:
            logger.warning(f"No camera mapping for {vf.name} — skipping")
            continue
        camera_id   = info["camera_id"]
        clip_start  = info["clip_start"]
        logger.info(f"\n▶  {vf.name}  →  {camera_id}  (clip_start={clip_start.isoformat()})")
        tracker.process_video_clip(
            str(vf), camera_id, clip_start,
            skip_frames=args.skip_frames,
        )

    all_events = tracker.get_events()
    emitter.emit_batch(all_events)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "events.jsonl"
    emitter.save_to_jsonl(str(out_file))

    stats = emitter.get_stats()
    logger.info("\n" + "=" * 50)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total events:    {stats['total_events']}")
    logger.info(f"Unique visitors: {stats['unique_visitors']}")
    logger.info(f"Staff events:    {stats['staff_events']}")
    logger.info(f"Duplicates:      {stats['duplicate']}")
    logger.info(f"Invalid dropped: {stats['invalid']}")
    logger.info("Event breakdown:")
    for etype, count in sorted(stats["event_types"].items()):
        logger.info(f"  {etype}: {count}")
    logger.info(f"\n✓ Events saved → {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())