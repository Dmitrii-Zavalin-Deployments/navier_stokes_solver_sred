#!/bin/bash
# 📤 Dropbox Upload Orchestrator — Updated for src/io/ architecture
# Final Phase of the Navier-Stokes Pipeline: Archiving & Cloud Export.

# 1. Environment Guard (From GitHub Secrets)
APP_KEY="${APP_KEY}"
APP_SECRET="${APP_SECRET}"
REFRESH_TOKEN="${REFRESH_TOKEN}"

# 2. Path Definition (SSoT)
# Ensure we use the workspace root provided by GitHub Actions
BASE_WORK_DIR="${GITHUB_WORKSPACE:-$(pwd)}"
DATA_DIR="${BASE_WORK_DIR}/data/testing-input-output"
ZIP_FILE_NAME="navier_stokes_output.zip"
LOCAL_ZIP_PATH="${DATA_DIR}/${ZIP_FILE_NAME}"

# 3. Archive Creation
# We zip the contents of the solver output directory created by Step 5
echo "📦 Finalizing Archive: ${ZIP_FILE_NAME}"

if [ -d "${DATA_DIR}/navier-stokes-output" ]; then
    (
      cd "${DATA_DIR}/navier-stokes-output" || exit 1
      zip -r -j "${ZIP_FILE_NAME}" ./*
      mv "${ZIP_FILE_NAME}" "${LOCAL_ZIP_PATH}"
    )
else
    echo "❌ ERROR: Solver output directory not found. Simulation may have failed."
    exit 1
fi

# 4. Confirm ZIP Integrity
if [ ! -f "${LOCAL_ZIP_PATH}" ]; then
    echo "❌ ERROR: Failed to create ZIP archive at ${LOCAL_ZIP_PATH}"
    exit 1
fi

# 5. Cloud Export Execution
# Define Dropbox destination
DROPBOX_DEST_FOLDER="/engineering_simulations_pipeline"

# Set PYTHONPATH so src.io.upload_to_dropbox can find src.dropbox_utils
export PYTHONPATH="${PYTHONPATH}:${BASE_WORK_DIR}"

echo "🔄 Uploading to Dropbox: ${DROPBOX_DEST_FOLDER}/${ZIP_FILE_NAME}"

python3 "${BASE_WORK_DIR}/src/io/upload_to_dropbox.py" \
    "${LOCAL_ZIP_PATH}" \
    "${DROPBOX_DEST_FOLDER}" \
    "${REFRESH_TOKEN}" \
    "${APP_KEY}" \
    "${APP_SECRET}"

# 6. Final Result Audit
if [ $? -eq 0 ]; then
    echo "✅ PIPELINE COMPLETE: ${ZIP_FILE_NAME} is now in the cloud."
else
    echo "❌ CRITICAL ERROR: Dropbox upload failed at the final gate."
    exit 1
fi