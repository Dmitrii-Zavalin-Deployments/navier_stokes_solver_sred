#!/bin/bash
echo "============================================================"
echo "🎯 TARGETED REPAIR: ELASTICITY LOG ALIGNMENT"
echo "============================================================"

# --- [1] Diagnostic: Trace the Elasticity Manager ---
echo "--- [Audit 1] Checking Elasticity Manager Logger Identity ---"
# We need to see if the elasticity logic uses 'Solver.Main' or its own name.
grep -r "getLogger" src/common/elasticity.py

# --- [2] Diagnostic: Smoking-Gun Source Audit ---
echo "--- [Audit 2] Verifying 'Instability' log string in source ---"
# Check the exact wording and the logger object being used.
cat -n src/common/elasticity.py | grep -C 5 "logger."

# --- [3] Diagnostic: Verify Propagation Chain ---
echo "--- [Audit 3] Checking if common/elasticity.py defines its own logger ---"
# If it defines 'logger = logging.getLogger(__name__)', it won't hit 'Solver.Main'
# unless it propagates to root or is a child (e.g., 'Solver.Main.Elasticity').
head -n 25 src/common/elasticity.py

# --- [4] AUTOMATED REPAIRS (Injections) ---

# REPAIR A: Force Elasticity to use the SSoT Logger (Solver.Main)
# This ensures all lifecycle events are captured by the test's listener.
# sed -i 's/getLogger(__name__)/getLogger("Solver.Main")/g' src/common/elasticity.py

# REPAIR B: Expand Test Listener (Alternative)
# If we want the test to be broader, we listen to the root, but we must ensure propagation is True everywhere.
# sed -i 's/logger="Solver.Main"/logger=""/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR C: Fix potential typo in the test assertion
# If the log actually says "Unstable" instead of "Instability", the test fails.
# sed -i 's/"Instability"/"Unstable"/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Forensic Audit complete. Review Audit 1 & 2 to choose Repair A or C."