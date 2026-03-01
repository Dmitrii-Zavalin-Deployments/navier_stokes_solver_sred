#!/bin/bash
# src/upload_to_dropbox.sh
# 📤 Dropbox Upload Orchestrator — Professional Cloud Export Gate

# 1. Environment Guard (Secrets from GitHub)
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# 2. Input Logic (Handshake from Point 2)
# If an argument is passed (the ZIP path), we use it. 
# Otherwise, we default to the standard location.
BASE_WORK_DIR="${GITHUB_WORKSPACE:-$(pwd)}"
DEFAULT_ZIP="${BASE_WORK_DIR}/data/testing-input-output/navier_stokes_output.zip"
LOCAL_ZIP_PATH="${1:-$DEFAULT_ZIP}"

echo "🔍 Validating archive for upload: $LOCAL_ZIP_PATH"

# 3. Explicit Guard (Zero-Debt Mandate)
if [ ! -f "$LOCAL_ZIP_PATH" ]; then
    echo "❌ ERROR: Target file not found at $LOCAL_ZIP_PATH."
    echo "Possible cause: Solver failed or path handshake was lost."
    exit 1
fi

# 4. Cloud Export Execution
DROPBOX_DEST_FOLDER="/engineering_simulations_pipeline"

# Set PYTHONPATH so the worker can find src.io.dropbox_utils
export PYTHONPATH="${PYTHONPATH}:${BASE_WORK_DIR}"

echo "🔄 Triggering Python Worker..."

python3 "${BASE_WORK_DIR}/src/io/upload_to_dropbox.py" \
    "$LOCAL_ZIP_PATH" \
    "$DROPBOX_DEST_FOLDER" \
    "$REFRESH_TOKEN" \
    "$APP_KEY" \
    "$APP_SECRET"

# 5. Final Result Audit
if [ $? -eq 0 ]; then
    echo "✅ PIPELINE COMPLETE: Upload successful."
else
    echo "❌ CRITICAL ERROR: Dropbox upload failed."
    exit 1
fi