#!/bin/bash
# src/download_from_dropbox.sh
# 📦 Ingestion Orchestrator — Bridges Dropbox Cloud to the Solver local state.

# 1. Environment Guard
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# 2. Path Definition (SSoT)
DROPBOX_FOLDER="/engineering_simulations_pipeline"
LOCAL_FOLDER="./data/testing-input-output"
LOG_FILE="./dropbox_download_log.txt"

# 3. Setup
echo "📂 Preparing ingestion workspace at $LOCAL_FOLDER..."
mkdir -p "$LOCAL_FOLDER"

# 4. Execution Logic
# Ensure Python can resolve 'src.io.dropbox_utils'
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🔄 Triggering Python Ingestion Worker (src/io/download_from_dropbox.py)..."

python3 src/io/download_from_dropbox.py \
    "$DROPBOX_FOLDER" \
    "$LOCAL_FOLDER" \
    "$REFRESH_TOKEN" \
    "$APP_KEY" \
    "$APP_SECRET" \
    "$LOG_FILE"

# 5. Result Verification
if [ $? -eq 0 ] && [ "$(ls -A "$LOCAL_FOLDER")" ]; then
    echo "✅ SUCCESS: Inputs synchronized. Ready for Solver execution."
    if [ -f "$LOG_FILE" ]; then
        echo "📜 Log Summary:"
        tail -n 3 "$LOG_FILE"
    fi
else
    echo "❌ ERROR: No input files found or sync failed. Pipeline halted."
    exit 1
fi