#!/bin/bash
echo "============================================================"
echo "🎯 FORENSIC AUDIT: THE SILENT OVERFLOW MYSTERY"
echo "============================================================"

# --- [1] Audit: Stack Trace & Exception Trap ---
echo "--- [Audit 1] Checking if error is swallowed in Step 3/4 ---"
# Rule 2: 100% visibility. Checking if sub-orchestrators have silent try/excepts.
grep -r "except:" src/step3 src/step4
grep -r "except Exception:" src/step3 src/step4

# --- [2] Audit: Logger Propagation ---
echo "--- [Audit 2] Verifying Logger Parentage ---"
# Rule 4: Hierarchy over Convenience. 
# If 'Solver.Main' doesn't propagate to root, caplog (defaulting to root) sees nothing.
grep -n "propagate" src/main_solver.py

# --- [3] Audit: The Smoking Gun (Step 3 Kernels) ---
echo "--- [Audit 3] Checking if advection/laplacian handles NaNs internally ---"
cat -n src/step3/ops/advection.py | head -n 50

# --- [4] AUTOMATED REPAIRS & TRAPS ---

# REPAIR A: Force Caplog to listen to the specific Logger Identity
# Rule 6: Efficiency. We tell the test exactly which pipe to listen to.
# sed -i 's/with caplog.at_level(logging.WARNING, logger=""):/with caplog.at_level(logging.WARNING, logger="Solver.Main"):/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Global NumPy Error Trap
# Rule 5: Deterministic Initialization. 
# We move the seterr to the very top of run_solver to ensure it's not overridden.
# sed -i '56i \    import numpy as np; np.seterr(all="raise")' src/main_solver.py

# REPAIR C: Broaden Exception Catching
# If it's not an ArithmeticError, it might be a RuntimeWarning being promoted to Error.
# sed -i 's/except (ArithmeticError, FloatingPointError, ValueError):/except (Exception):/g' src/main_solver.py

# REPAIR D: Direct STDOUT Signal
# Rule 7: Granular Traceability. We bypass the logger for a moment to prove the path exists.
# sed -i '125i \                print("STABILITY_SIGNAL_REACHED")' src/main_solver.py

echo "============================================================"
echo "✅ Audit Block Configured. Run to expose the silent failure."