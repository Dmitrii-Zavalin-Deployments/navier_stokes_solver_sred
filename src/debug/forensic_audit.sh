#!/bin/bash

# Configuration
TARGET="config.json"
LOG_FILE="forensic_trace.log"
TIMESTAMP=$(date +"%H:%M:%S")

echo "--- [FORENSIC CHECK @ $TIMESTAMP] ---" | tee -a $LOG_FILE

if [ ! -f "$TARGET" ]; then
    echo "❌ FATAL: $TARGET has been physically DELETED from the disk!" | tee -a $LOG_FILE
    ls -la | tee -a $LOG_FILE
    exit 1
fi

# Check for the key ppe_max_retries
if grep -q "ppe_max_retries" "$TARGET"; then
    echo "✅ Key 'ppe_max_retries' is present." | tee -a $LOG_FILE
    # Show the line for absolute certainty
    grep "ppe_max_retries" "$TARGET" | tee -a $LOG_FILE
else
    echo "❌ CRITICAL: Key 'ppe_max_retries' is MISSING!" | tee -a $LOG_FILE
    echo "Current File Content:" | tee -a $LOG_FILE
    cat "$TARGET" | tee -a $LOG_FILE
    
    # Traceability: Who was the last user/process to touch it?
    echo "File Metadata:" | tee -a $LOG_FILE
    stat "$TARGET" | tee -a $LOG_FILE
fi

echo "--------------------------------------" | tee -a $LOG_FILE

# ==============================================================================
# 1. FIND THE KILLER (Grep for write/dump operations in Python)
# ==============================================================================
echo "--- [SEARCHING FOR IO OVERWRITES] ---"
grep -rE "json\.dump|open.*'w'|write\(.*json" src/

# ==============================================================================
# 2. AUDIT THE SMOKING GUNS (Viewing the logic)
# ==============================================================================
echo -e "\n--- [SMOKING GUN 1: The Config Loader] ---"
# We check if the __init__ is missing the retries assignment (Trace 4 confirmed this)
cat -n src/common/solver_config.py | head -n 35

echo -e "\n--- [SMOKING GUN 2: The Logic that Saves to Disk] ---"
# Searching for the method that actually touches config.json
grep -nC 5 "config.json" src/common/solver_config.py 2>/dev/null || echo "Not found in solver_config.py"

# # ==============================================================================
# # 3. THE REPAIR (Sed Injections)
# # ==============================================================================
# echo -e "\n--- [APPLYING PERMANENT FIXES] ---"

# # FIX A: Inject the missing attribute into the Constructor so it survives in memory
# # We look for the ppe_max_iter line and append the ppe_max_retries after it.
# sed -i '/self.ppe_max_iter =/a \        self.ppe_max_retries = kwargs.get("ppe_max_retries", 10)' src/common/solver_config.py

# # FIX B: (Hypothetical but likely) If there is a to_dict() or save() method, 
# # ensure ppe_max_retries is included. We'll run a broad injection to ensure
# # the property exists for any serialization.
# # Check if there's a dictionary mapping and add it if missing.
# if grep -q "ppe_max_iter" src/common/solver_config.py; then
#     sed -i 's/"ppe_max_iter": self.ppe_max_iter/"ppe_max_iter": self.ppe_max_iter, "ppe_max_retries": self.ppe_max_retries/g' src/common/solver_config.py
# fi

# # FIX C: Fix the Audit script "1010" bug so it doesn't fail even if data is duplicated
# sed -i 's/EXPECTED_RETRIES=.*/EXPECTED_RETRIES=$(grep "ppe_max_retries" "$CONFIG_FILE" | head -n 1 | grep -oE "[0-9]+")/' src/debug/forensic_audit.sh

# ==============================================================================
# 4. FINAL VERIFICATION
# ==============================================================================
echo -e "\n--- [VERIFYING REPAIR] ---"
cat -n src/common/solver_config.py | grep -C 2 "ppe_max_retries"