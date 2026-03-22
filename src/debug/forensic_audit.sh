echo "============================================================"
echo "🔍 DIAGNOSING LOG PROPAGATION & CAPTURE"
echo "============================================================"

# --- [1] Diagnostic: Check Logger Propagation ---
echo "--- [Audit 1] Checking if Solver.Main propagates to Root ---"
# If propagation is False, caplog (listening at root) will never see the records.
grep -n "propagate" src/main_solver.py || echo "Propagation not explicitly set (Default is True, but worth checking)."

# --- [2] Diagnostic: Verify Log Content ---
echo "--- [Audit 2] Exact Log String Verification ---"
# Ensure the string "Instability" hasn't been typo'd in the source.
cat -n src/main_solver.py | grep "logger.warning"

# --- [3] Diagnostic: Inspect Test Capture ---
echo "--- [Audit 3] Test File Listener Scope ---"
# Checking how many times caplog.at_level appears and if it specifies a logger.
grep -n "caplog.at_level" tests/property_integrity/test_heavy_elasticity_lifecycle.py

# --- [4] AUTOMATED REPAIRS (Injections) ---

# REPAIR A: Update all caplog contexts in the test file to listen specifically to 'Solver.Main'.
# This is the most surgical fix to align the test with the Rule 4 implementation.
# sed -i 's/caplog.at_level(logging.WARNING):/caplog.at_level(logging.WARNING, logger="Solver.Main"):/g' tests/property_integrity/test_heavy_elasticity_lifecycle.py

# REPAIR B: Ensure propagation is enabled at the module level.
# This forces the named logger to send its records up to the root where caplog sits.
# sed -i '/logger = logging.getLogger("Solver.Main")/a logger.propagate = True' src/main_solver.py

# REPAIR C: Fix Rule 5 edge case. 
# In some environments, np.seterr(all="raise") requires a reset before being set again.
# sed -i '/np.seterr(all="raise", under="ignore")/i \    np.seterr(all="warn")' src/main_solver.py

echo "============================================================"
echo "✅ Forensic Audit and Repair Script Ready"