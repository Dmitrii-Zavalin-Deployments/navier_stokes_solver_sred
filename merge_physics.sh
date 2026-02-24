#!/bin/bash

# Define the target file for the Step 1 Orchestration group
TARGET="tests/step1/test_step1_orchestration.py"

echo "Consolidating Step 1 Orchestration and Pipeline tests..."

# 1. Start with the Main Orchestrator (includes primary imports and fixtures)
cat tests/step1/test_orchestrate_step1.py > $TARGET

# 2. Append Initial Condition (IC) Exceptions
echo -e "\n\n# --- SECTION: Initial Condition (IC) Exceptions ---" >> $TARGET
grep -vE "^import |^from " tests/step1/test_ic_exceptions.py >> $TARGET

# 3. Append Step 1 Edge Cases
echo -e "\n\n# --- SECTION: Step 1 Edge Cases ---" >> $TARGET
grep -vE "^import |^from " tests/step1/test_step1_edge_cases.py >> $TARGET

# 4. Append Data Coverage (Loud Value Traceability)
echo -e "\n\n# --- SECTION: Step 1 Data Coverage ---" >> $TARGET
grep -vE "^import |^from " tests/step1/test_step1_data_coverage.py >> $TARGET

# 5. Append Output Dummy Schema Compliance
echo -e "\n\n# --- SECTION: Output Schema Compliance ---" >> $TARGET
grep -vE "^import |^from " tests/step1/test_step1_output_dummy_schema.py >> $TARGET

echo "Successfully created $TARGET"

# Final verification of the orchestration suite
ls -lh $TARGET