#!/bin/bash
# 📦 Dropbox Sync Script — Updated for src/io/ architecture
# Enforces the Phase C "Explicit or Error" mandate for Cloud I/O.

# 1. Environment Guard
# These are populated by GitHub Actions secrets via the .yml file
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# 2. Path Definition (SSoT)
DROPBOX_FOLDER="/engineering_simulations_pipeline"
LOCAL_FOLDER="./data/testing-input-output"
LOG_FILE="./dropbox_download_log.txt"

# 3. Setup
echo "📂 Initializing local workspace at $LOCAL_FOLDER..."
mkdir -p "$LOCAL_FOLDER"

# 4. Execution
# We use -m to run the module correctly or set PYTHONPATH to ensure
# that 'src.dropbox_utils' remains resolvable from the root.
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🔄 Invoking Dropbox Python Worker (src/io/download_dropbox_files.py)..."

python3 src/io/download_dropbox_files.py \
    "$DROPBOX_FOLDER" \
    "$LOCAL_FOLDER" \
    "$REFRESH_TOKEN" \
    "$APP_KEY" \
    "$APP_SECRET" \
    "$LOG_FILE"

# 5. Audit Trace & Validation
if [ -f "$LOG_FILE" ]; then
    echo "📜 Worker Log snippet:"
    tail -n 5 "$LOG_FILE"
fi

if [ "$(ls -A "$LOCAL_FOLDER")" ]; then
    echo "✅ SUCCESS: Files successfully synchronized to $LOCAL_FOLDER"
else
    echo "❌ ERROR: Local folder is empty. Dropbox sync failed or no valid files found."
    exit 1
fi