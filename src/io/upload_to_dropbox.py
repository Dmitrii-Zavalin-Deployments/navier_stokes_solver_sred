# src/io/upload_to_dropbox.py

#!/bin/bash
# 📤 Dropbox Upload Orchestrator
# Accepts the ZIP path directly from the GitHub Environment.

# 1. Environment Guard
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# 2. Input Argument Guard (Phase C: Explicit or Error)
# The path is passed from the .yml file as $1
LOCAL_ZIP_PATH="$1"

if [ -z "$LOCAL_ZIP_PATH" ]; then
    echo "❌ ERROR: No archive path provided to the upload script."
    exit 1
fi

if [ ! -f "$LOCAL_ZIP_PATH" ]; then
    echo "❌ ERROR: Target file not found at $LOCAL_ZIP_PATH"
    exit 1
fi

# 3. Cloud Export
DROPBOX_DEST_FOLDER="/engineering_simulations_pipeline"
export PYTHONPATH="${PYTHONPATH}:${GITHUB_WORKSPACE}"

echo "🔄 Uploading to Dropbox: $LOCAL_ZIP_PATH"

python3 "src/io/upload_to_dropbox.py" \
    "$LOCAL_ZIP_PATH" \
    "$DROPBOX_DEST_FOLDER" \
    "$REFRESH_TOKEN" \
    "$APP_KEY" \
    "$APP_SECRET"

# 4. Final Result Audit
if [ $? -eq 0 ]; then
    echo "✅ PIPELINE COMPLETE: Upload successful."
else
    echo "❌ CRITICAL ERROR: Dropbox upload failed."
    exit 1
fi