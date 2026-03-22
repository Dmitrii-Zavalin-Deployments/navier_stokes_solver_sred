#!/bin/bash
# src/debug/forensic_audit.sh

echo "============================================================"
echo "🔍 STARTING DEEP FORENSIC AUDIT: THE LOGGING & OVERFLOW GHOST"
echo "============================================================"

# --- [1] Verify Logger Level in ElasticManager ---
echo "--- [Audit 1] Checking ElasticManager Logging Level ---"
grep -C 2 "Instability" src/common/elasticity.py | grep "self.logger"

# --- [2] Source Audit: main_solver.py loop integrity ---
echo "--- [Audit 2] Verification of Exception Catch Block ---"
cat -n src/main_solver.py | sed -n '120,130p'

# --- [3] Repair: Ensure ElasticManager uses WARNING for instabilities ---
# Rule 7 requires high-resolution trace for recovery. 
# If it's currently .info(), the test won't see it via caplog.at_level(logging.WARNING).
# sed -i 's/self.logger.info(f"Instability/self.logger.warning(f"Instability/g' src/common/elasticity.py

# --- [4] Repair: Fix main_solver.py Logger Propagation ---
# Ensure the main_solver's catch block actually triggers a log entry that pytest can see.
# sed -i '123i \                logger.warning("Instability detected: triggering recovery path")' src/main_solver.py

# --- [5] Rule 5 Compliance: Double-check the 'raise' trigger ---
echo "--- [Audit 3] Checking for potential silent NaN suppressors ---"
grep -r "np.seterr" src/

echo "✅ Audit Commands Ready."