#!/bin/bash
echo "============================================================"
echo "🎯 PHASE C: NUMERICAL TRAP & STABILITY AUDIT"
echo "============================================================"

# --- [Audit 1] Runtime Configuration Check ---
echo "--- [Audit 1] Verifying NumPy Error Traps in Main Solver ---"
# Rule 5: Check if np.seterr is actually being called
grep -n "np.seterr" src/main_solver.py

# --- [Audit 2] Source Audit: The Try/Except Trap ---
echo "--- [Audit 2] Inspecting the Recovery Logic Block ---"
cat -n src/main_solver.py | sed -n '120,150p'

# --- [Audit 3] Data Forensic: Inspecting the 'Successful' Failure ---
echo "--- [Audit 3] Checking for NaNs in the generated test artifacts ---"
# If the test passed but should have failed, the HDF5 likely contains Infs/NaNs
python3 -c "import h5py; import numpy as np; f = h5py.File('navier_stokes_output/snapshot_0001.h5', 'r'); print('VX Max:', np.max(f['vx'][:])); print('Contains NaN:', np.isnan(f['vx'][:]).any())"

# --- [Audit 4] Elasticity Logic Check ---
echo "--- [Audit 4] Verifying ElasticityManager stabilization trigger ---"
cat -n src/common/elasticity.py | grep -A 5 "def stabilization"

# --- [5] AUTOMATED REPAIRS (Candidate Injections) ---

# REPAIR A: Force a 'Zero Tolerance' trap for all floating point anomalies
# This ensures that even tiny overflows trigger the recovery path.
# sed -i 's/np.seterr(all="raise", under="ignore")/np.seterr(all="raise")/g' src/main_solver.py

# REPAIR B: Inject an explicit NaN check inside the main loop to force recovery
# This manually triggers the recovery if NumPy's hardware trap fails.
# sed -i '/state.time += elasticity.dt/a \            if not np.isfinite(state.time): raise ValueError("Time Instability")' src/main_solver.py

# REPAIR C: Increase test duration to allow instability to manifest
# sed -i 's/"total_time": 0.2/"total_time": 1.0/' tests/property_integrity/test_heavy_elasticity_lifecycle.py

echo "============================================================"
echo "✅ Audit Complete. Check Audit 3 for 'Contains NaN: True'."