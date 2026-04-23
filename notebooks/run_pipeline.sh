#!/usr/bin/env bash
# run_pipeline.sh
# Waits for any in-flight uwbLSTMCNN_v2.py to finish, then runs:
#   1. fusionE2E.py   (joint lip fine-tuning)
#   2. fusionGate.py  (final fusion evaluation)
#
# Usage:  bash run_pipeline.sh [UWB_PID]
# If UWB_PID is given, the script waits for that process to exit first.

set -e
cd "$(dirname "$0")"
LOG_DIR="$(pwd)/pipeline_logs"
mkdir -p "$LOG_DIR"

ts() { date '+%H:%M:%S'; }

# ── 0. Extract UWB embeddings from saved checkpoint ──────────────────────────
echo "[$(ts)] ===== Step 0: UWB embedding extraction ====="
python3 uwb_extract_embeddings.py 2>&1 | tee "$LOG_DIR/uwb_extract.log"
echo "[$(ts)] UWB extraction complete."

# ── 1. fusionE2E.py ──────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] ===== Step 1: fusionE2E.py ====="
python3 fusionE2E.py 2>&1 | tee "$LOG_DIR/fusionE2E.log"
echo "[$(ts)] fusionE2E.py complete."

# ── 2. fusionGate.py ─────────────────────────────────────────────────────────
echo ""
echo "[$(ts)] ===== Step 2: fusionGate.py ====="
python3 fusionGate.py 2>&1 | tee "$LOG_DIR/fusionGate.log"
echo "[$(ts)] fusionGate.py complete."

echo ""
echo "[$(ts)] Pipeline finished. Logs in $LOG_DIR/"