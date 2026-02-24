#!/bin/bash

# Define the target file
TARGET="tests/step1/test_config_and_physics.py"

echo "Consolidating Config and Physics tests..."

# 1. Start with the main config file (includes necessary imports)
cat tests/step1/test_physics_and_config.py > $TARGET

# 2. Append Derived Constants (Base Logic)
echo -e "\n\n# --- SECTION: Derived Constants ---" >> $TARGET
# Filter out the imports to avoid clutter
grep -vE "^import |^from " tests/step1/test_compute_derived_constants.py >> $TARGET

# 3. Append Derived Constants (Math/Edge Cases)
echo -e "\n\n# --- SECTION: Derived Constants Math ---" >> $TARGET
# Filter out the imports
grep -vE "^import |^from " tests/step1/test_compute_derived_constants_math.py >> $TARGET

echo "Successfully created $TARGET"

# Optional: verify the file exists
ls -l $TARGET
