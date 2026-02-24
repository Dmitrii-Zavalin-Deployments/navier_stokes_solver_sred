#!/bin/bash

# Define the target file for the Grid and Allocation group
TARGET="tests/step1/test_grid_and_allocation.py"

echo "Consolidating Grid and Allocation tests..."

# 1. Start with Grid Allocation (contains necessary imports/fixtures)
# We use 'cat' for the first file to establish the file and its imports.
cat tests/step1/test_grid_allocation.py > $TARGET

# 2. Append Initialize Grid Full
echo -e "\n\n# --- SECTION: Initialize Grid Full ---" >> $TARGET
# Filter out imports to prevent 'import pytest' being repeated 20 times
grep -vE "^import |^from " tests/step1/test_initialize_grid_full.py >> $TARGET

# 3. Append Verify Shapes
echo -e "\n\n# --- SECTION: Verify Shapes (Cell-Centered) ---" >> $TARGET
grep -vE "^import |^from " tests/step1/test_verify_shapes.py >> $TARGET

echo "Successfully created $TARGET"

# Verify the new file structure
ls -lh $TARGET