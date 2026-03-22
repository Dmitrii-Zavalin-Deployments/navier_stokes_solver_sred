#!/bin/bash
echo "============================================================"
echo "🎯 FORENSIC AUDIT: TERMINATING THE SILENT SWALLOW"
echo "============================================================"

# --- [1] Audit: The "Swallow" Logic ---
echo "--- [Audit 1] Examining Step 3 Exception Handling ---"
# Checking if orchestrate_step3 is swallowing ArithmeticError
cat -n src/step3/orchestrate_step3.py | sed -n '40,60p'

# --- [2] Audit: Precision Drift (Rule 1) ---
echo "--- [Audit 2] Checking for float64 vs float32 accumulation ---"
# Rule 1 states float64 is for accumulation. If intermediate products are float32, 
# 1e15 * 1e15 will hit 'inf' instantly.
grep "dtype" src/common/field_schema.py

# --- [3] Audit: Logger Injection Point ---
echo "--- [Audit 3] Verifying if Main Solver logger is correctly initialized ---"
head -n 30 src/main_solver.py | grep "getLogger"

# --- [4] AUTOMATED REPAIRS ---

# REPAIR A: Force Rule 7 Compliance in Advection
# We insert a hard guard to ensure an exception is raised regardless of NumPy global state.
# sed -i '42i \        if not (np.isfinite(u_c) and np.isfinite(df_dx)): raise ArithmeticError("NaN/Inf detected in advection kernel")' src/step3/ops/advection.py

# REPAIR B: Ensure Exception Propagation in Orchestrator
# If Step 3 catches an error, it MUST re-raise it for the Elasticity Manager to see it.
# sed -i '/except /a \            raise' src/step3/orchestrate_step3.py

# REPAIR C: Force Logger Identity for the Test
# Standardizing the caplog capture to ensure we are listening to the right 'pipe'.
# sed -i 's/logger=""/logger="Solver.Main"/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR D: Deterministic Reset (Rule 5)
# Ensure dt reduction is actually printed to stdout for visual confirmation.
# sed -i '/logger.warning("Instability/a \            print(f"DEBUG: Reducing time step due to instability")' src/main_solver.py

echo "============================================================"
echo "✅ Forensic Script Ready. Run to expose the hidden 'except' block."