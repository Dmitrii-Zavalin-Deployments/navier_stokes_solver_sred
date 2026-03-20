echo "--- 1. THE NAMING SMOKING GUN ---"
# Check the test file to see what filename it expects
grep "integration_input.zip" tests/property_integrity/test_heavy_elasticity_lifecycle.py || echo "❌ Test is looking for a different filename than the archiver produces."

echo "--- 2. THE ARCHIVER FILENAME PROOF ---"
# Check the archiver to see what it actually names the file
grep "final_destination =" src/common/archive_service.py

echo "--- 3. LOCATE THE 'GHOST' ZIP ---"
# Find where the zip actually went during the last run
find . -name "*.zip"

echo "--- 4. SIMULATION FINAL STATE AUDIT ---"
# Prove the solver finished and reached the return statement
# We look for the 'ready_for_time_loop = False' trigger
grep -n "ready_for_time_loop = False" src/main_solver.py

echo "--- 5. DIRECTORY CONTENT AT TERMINATION ---"
# See if 'navier_stokes_output' directory was created before being zipped
ls -ld navier_stokes_output || echo "❌ Staging directory not found."