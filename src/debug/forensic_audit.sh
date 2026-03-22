echo "============================================================"
echo "🔍 DIAGNOSING MISSING INSTABILITY SIGNAL"
echo "============================================================"

# --- [1] Diagnostic: Check how main_solver handles exceptions ---
echo "--- [Audit 1] main_solver.py Exception Handling ---"
# We need to see if there is a logger.warning call with "Instability"
grep -n "except " src/main_solver.py -A 5

# --- [2] Diagnostic: Check Elasticity Manager usage ---
echo "--- [Audit 2] ElasticManager Signal check ---"
# See if the word 'Instability' exists anywhere in the source now
grep -r "Instability" src/

# --- [3] Smoking Gun: View the catch block in main_solver ---
echo "--- [Audit 3] main_solver.py Line Audit ---"
cat -n src/main_solver.py | sed -n '100,150p'

# --- [4] Automated Repairs (Drafts) ---

# REPAIR A: Inject the missing log signal into the main_solver catch block
# We assume the catch block is for (ArithmeticError, ValueError)
# # sed -i '/except (ArithmeticError, ValueError):/a \    import logging; logging.getLogger("Solver.Main").warning("Instability detected: numerical divergence triggered recovery path.")' src/main_solver.py

# REPAIR B: Ensure the logger name aligns with the test expectations
# # sed -i 's/logger.error(f"Error/logger.warning(f"Instability/g' src/main_solver.py

# REPAIR C: Force the Elasticity logic to log the keyword if main_solver is clean
# # sed -i '/def reduce_dt/a \    logging.getLogger("Solver.Elasticity").warning("Instability: reducing time-step to maintain CFL condition.")' src/common/elasticity.py

echo "============================================================"
echo "✅ Forensic Audit and Repair Script Ready"