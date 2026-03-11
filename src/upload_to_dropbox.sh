#!/bin/bash
# src/upload_to_dropbox.sh
# 📤 Dropbox Upload Orchestrator — Professional Cloud Export Gate.

# 1. Environment Guard
if [[ -z "${APP_KEY}" || -z "${APP_SECRET}" || -z "${REFRESH_TOKEN}" ]]; then
    echo "❌ ERROR: Missing required credentials."
    exit 1
fi

# 2. Path Resolution
BASE_WORK_DIR="${GITHUB_WORKSPACE:-$(pwd)}"
DEFAULT_ZIP="${BASE_WORK_DIR}/data/testing-input-output/navier_stokes_output.zip"
LOCAL_ZIP_PATH="${1:-$DEFAULT_ZIP}"

# 3. Validation (Zero-Debt Mandate)
if [ ! -f "$LOCAL_ZIP_PATH" ]; then
    echo "❌ ERROR: Target file not found at $LOCAL_ZIP_PATH."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:${BASE_WORK_DIR}"

echo "🔄 Triggering Python CloudUploader..."

# 4. Explicit Orchestration via Inline Python
# Bypasses CLI-argument reliance for safer, deterministic instantiation.
python3 -c "
from pathlib import Path
from src.io.dropbox_utils import TokenManager
from src.io.upload_to_dropbox import CloudUploader
import os

# Initialize components explicitly
tm = TokenManager(client_id=os.environ['APP_KEY'], client_secret=os.environ['APP_SECRET'])
uploader = CloudUploader(tm, os.environ['REFRESH_TOKEN'])

# Execute atomic upload
uploader.upload(Path(os.environ['LOCAL_ZIP_PATH']), '/engineering_simulations_pipeline')
"

# 5. Final Result Audit
if [ $? -eq 0 ]; then
    echo "✅ PIPELINE COMPLETE: Upload successful."
else
    echo "❌ CRITICAL ERROR: Dropbox upload failed."
    exit 1
fi