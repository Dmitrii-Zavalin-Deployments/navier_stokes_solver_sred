# 1. Structural Check: Verify file path
ls -R src/common/simulation_context.py

# 2. Character Audit: Inspect the "Useless Expression" at Line 27
cat -n src/common/simulation_context.py | sed -n '23,35p'

# 3. Target Verification: Ensure the SolverConfig constructor call is right below it
grep -A 3 "SolverConfig(" src/common/simulation_context.py

# 1. Remove the orphaned expression and the outdated comments (Lines 25-31)
# This clears the B018 error and adheres to Rule 2 (Zero-Debt).
sed -i '25,31d' src/common/simulation_context.py

# 2. Fix the SolverConfig initialization (Line 33 in original, now shifted)
# We remove 'dt=base_dt' because SolverConfig no longer has that slot.
sed -i 's/config = SolverConfig(dt=base_dt, \*\*config_dict)/config = SolverConfig(\*\*config_dict)/' src/common/simulation_context.py

# 3. Final atomic lint and fix to normalize spacing
ruff check src/common/simulation_context.py --fix

# 4. Success Signal
echo "✅ Ghost logic purged. SimulationContext now follows SSoT (Rule 4)."