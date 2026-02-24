#!/bin/bash

# Define the target file for the Topology and Boundaries group
TARGET="tests/step1/test_topology_and_boundaries.py"

echo "Consolidating Topology and Boundaries tests..."

# 1. Start with Boundary Conditions (establishes imports and fixtures)
cat tests/step1/test_boundary_conditions.py > $TARGET

# 2. Append State and Masking
echo -e "\n\n# --- SECTION: State and Masking (Geometry Mapping) ---" >> $TARGET
# Filter out imports to prevent 'import pytest' etc. being repeated
grep -vE "^import |^from " tests/step1/test_state_and_masking.py >> $TARGET

echo "Successfully created $TARGET"

# Verify the file size and existence
ls -lh $TARGET