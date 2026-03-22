#!/bin/bash
echo "============================================================"
echo "🎯 PHASE D: PROPAGATION DEPTH & EXCEPTION TRAP AUDIT"
echo "============================================================"

# --- [Audit 1] Iteration Count Check ---
echo "--- [Audit 1] Checking how many steps actually ran ---"
# If we see 3 flushes but 0 warnings, the instability isn't 'blooming' fast enough
grep -c "Explicitly flushing" logs/solver.log || echo "No log file found, check stdout."

# --- [Audit 2] Numerical Kernel Audit ---
echo "--- [Audit 2] Checking if Step 3 (PPE) is silently returning NaNs ---"
# Rule 7: Bubbling. If PPE returns NaN but doesn't RAISE, the loop continues blindly.
grep -n "return.*nan" src/step3/ppe_solver.py

# --- [Audit 3] Boundary Condition (BC) Latency ---
echo "--- [Audit 3] Checking if BCs are applied BEFORE or AFTER the first solve ---"
cat -n src/main_solver.py | grep -A 5 "orchestrate_step4"

# --- [Audit 4] Logger Propagation ---
echo "--- [Audit 4] Checking if logger level is being overridden at runtime ---"
grep "setLevel" src/main_solver.py

# --- [5] AUTOMATED REPAIRS (Force the Failure) ---

# REPAIR A: Increase simulation duration to ensure divergence
# sed -i 's/"total_time": 0.2/"total_time": 5.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Force a ValueError if velocity exceeds a physical constant (CFL trigger)
# sed -i '/orchestrate_step3/a \                if np.max(np.abs(block.field.u)) > 1e10: raise ValueError("Physical Divergence")' src/main_solver.py

# REPAIR C: Ensure the logger name in the test matches the one in Elasticity
# sed -i 's/logger="Solver.Main"/logger="Solver"/' tests/property_integrity/test_heavy_elasticity_lifecycle.py
# sed -i 's/getLogger("Solver.Main")/getLogger("Solver")/' src/common/elasticity.py

echo "============================================================"
echo "✅ Audit Complete. Use REPAIR A to give the instability time to bloom."