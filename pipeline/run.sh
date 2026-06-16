#!/usr/bin/env bash
# pipeline/run.sh — One command to process all clips → events.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE="${ROOT}/pipeline"
VIDEOS="${ROOT}/videos"
OUTPUT="${ROOT}/events"
LAYOUT="${ROOT}/store_layout.json"
POS="${ROOT}/pos_transactions.csv"
if [[ ! -f "${POS}" ]]; then
    POS="${ROOT}/pos_transactions.example.csv"
fi

echo "Store Intelligence Detection Pipeline"
echo "======================================"
echo "Videos : ${VIDEOS}"
echo "Output : ${OUTPUT}"
echo ""

python "${PIPELINE}/run_detection.py" \
    --videos-dir           "${VIDEOS}"  \
    --output-dir           "${OUTPUT}"  \
    --store-layout         "${LAYOUT}"  \
    --pos-csv              "${POS}"     \
    --model                yolov8n.pt   \
    --confidence-threshold 0.4          \
    --skip-frames          2

echo ""
echo "✓ Done. Events → ${OUTPUT}/events.jsonl"
echo "  Line count: $(wc -l < "${OUTPUT}/events.jsonl")"