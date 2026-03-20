echo "--- 1. IDENTIFYING ALL MOCK_CONFIG LOCATIONS ---"
grep -r "MOCK_CONFIG =" tests/helpers/

echo "--- 2. SMOKING GUN: SOLVER_CONFIG INITIALIZATION LOGIC ---"
cat -n src/common/solver_config.py | head -n 45

echo "--- 3. DATA PERSISTENCE CHECK ---"
# Verifying if any archives were actually generated despite the setup errors
ls -l data/testing-input-output/

echo "--- 4. SURGICAL INJECTION A: REPAIR TEST HELPERS ---"
# Add the missing field to the primary mock helper
sed -i '/"ppe_omega": 1.7/a \    "divergence_threshold": 1e6,' tests/helpers/solver_step5_output_dummy.py

echo "--- 5. SURGICAL INJECTION B: REPAIR TEST FIXTURES ---"
# Fix the direct instantiation in the property integrity tests
sed -i 's/ppe_omega=1.0/ppe_omega=1.0, divergence_threshold=1e6/g' tests/property_integrity/test_step1_initialization.py
sed -i 's/ppe_omega=1.0/ppe_omega=1.0, divergence_threshold=1e6/g' tests/property_integrity/test_step3_initialization.py
sed -i 's/ppe_omega=1.0/ppe_omega=1.0, divergence_threshold=1e6/g' tests/property_integrity/test_step5_initialization.py

echo "--- 6. VERIFYING REPAIR ---"
grep "divergence_threshold" tests/helpers/solver_step5_output_dummy.py