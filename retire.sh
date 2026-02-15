#!/usr/bin/env bash
set -euo pipefail

# 1. Ensure we are in the project root
#    (adjust if your root folder name is different)
if [ ! -d "tests" ] || [ ! -d "schema" ]; then
  echo "Please run this script from the project root (navier_stokes_solver_sred)."
  exit 1
fi

echo "=== Creating archive folders for per-step schema tests ==="
mkdir -p tests/step1/archive
mkdir -p tests/step2/archive
mkdir -p tests/step3/archive
mkdir -p tests/step4/archive

echo "=== Moving per-step schema tests into archive/ ==="
if [ -f tests/step1/test_step1_schema_output.py ]; then
  mv tests/step1/test_step1_schema_output.py tests/step1/archive/
fi

if [ -f tests/step2/test_step2_schema_output.py ]; then
  mv tests/step2/test_step2_schema_output.py tests/step2/archive/
fi

if [ -f tests/step3/test_step3_schema_output.py ]; then
  mv tests/step3/test_step3_schema_output.py tests/step3/archive/
fi

if [ -f tests/step4/test_step4_schema_output.py ]; then
  mv tests/step4/test_step4_schema_output.py tests/step4/archive/
fi

echo "=== Removing per-step schema JSON files (Step 1–4) ==="
# Comment out the next block if you want to keep the files for a while longer.
for f in \
  schema/step1_output_schema.json \
  schema/step2_output_schema.json \
  schema/step3_output_schema.json \
  schema/step4_output_schema.json
do
  if [ -f "$f" ]; then
    git rm "$f"
  fi
done

echo "=== Reminder: per-step validation calls must be removed in code manually ==="
echo "  - Step 1: src/step1/orchestrate_step1.py"
echo "  - Step 2: src/step2/orchestrate_step2.py"
echo "  - Step 3: src/step3/orchestrate_step3.py"
echo "  - Step 4: (if any) src/step4/orchestrate_step4.py"
echo "Add a short DEPRECATED comment where you remove them."

echo "=== Running tests (optional but recommended) ==="
pytest

echo "=== Done. Per-step schemas retired; final schema is now the single contract. ==="

