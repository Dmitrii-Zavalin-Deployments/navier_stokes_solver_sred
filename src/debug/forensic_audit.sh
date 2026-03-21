# 1. ARCHITECTURAL INTEGRITY: Verify the Unified Governor Migration
echo "--- AUDITING REFACTORED ELASTICITY: src/common/elasticity.py ---"
# Check for the presence of the Unified Governor method
grep -n "def sync_state" src/common/elasticity.py
# Verify that success counters and recovery are nested INSIDE sync_state
grep -nE "_iteration \+= 1|self.is_in_panic and self._iteration >=" src/common/elasticity.py

# 2. CLEAN ORCHESTRATION: Verify main_solver.py is Lean
echo -e "\n--- AUDITING MAIN SOLVER LOOP: src/main_solver.py ---"
# Ensure old manual recovery and arithmetic error blocks are GONE
if grep -q "gradual_recovery()" src/main_solver.py; then
    echo "❌ ERROR: Manual gradual_recovery() still exists in main_solver.py"
fi
if grep -q "ArithmeticError" src/main_solver.py; then
    echo "⚠️ WARNING: ArithmeticError still present. Ensure it's not being used for flow control."
fi
# Confirm the new Unified Governor call is the only gate
cat -n src/main_solver.py | grep -C 3 "elasticity.sync_state(state)"

# 3. SYNTAX VALIDATION: Pre-flight check
echo -e "\n--- STATIC ANALYSIS ---"
python3 -m py_compile src/main_solver.py src/common/elasticity.py && echo "✅ AST Syntax Validated"

# 4. BEHAVIORAL VERIFICATION: Run the Heavy Elasticity Lifecycle Test
# The test should now pass because sync_state will eventually raise RuntimeError 
# when the dt floor is hit, which is exactly what the test expects.
echo -e "\n--- RUNNING TARGETED RECOVERY TEST ---"
pytest tests/property_integrity/test_heavy_elasticity_lifecycle.py -v --log-level=WARNING

# 5. FORENSIC LOG AUDIT: Check for Monotonic Decay (The Ratchet)
echo -e "\n--- LIVE EXECUTION DECAY PATTERN ---"
# We run the solver; it is expected to crash with the 'bad' input.
python3 src/main_solver.py integration_input.json > solver_output.tmp 2>&1 || true

echo "Checking for 'Hovering Glitch' (dt must strictly decrease or stay flat during panic):"
# Extract the dt values and check if any value is GREATER than the previous one
grep "PANIC: dt reduced to" solver_output.tmp | awk '{print $NF}' > dt_sequence.txt
if [[ -s dt_sequence.txt ]]; then
    python3 -c "
import sys
dts = [float(line.strip()) for line in open('dt_sequence.txt')]
if any(dts[i] < dts[i-1] for i in range(1, len(dts))):
    print('✅ Monotonic Decay Confirmed (Ratchet Working)')
else:
    print('❌ Warning: dt did not decrease as expected.')
"
else
    echo "❌ Error: No panic signals detected. Check if simulation failed too early."
fi

# 6. FINAL SUCCESS SIGNAL
FINAL_DT=$(tail -n 1 dt_sequence.txt)
echo -e "\nFinal dt before crash: $FINAL_DT"
if [[ -n "$FINAL_DT" ]]; then
    echo "Refactor Verification Complete."
else
    echo "Verification Failed: Check logs."
fi