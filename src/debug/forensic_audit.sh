echo "--- 1. HUNTING THE .ZIP.ZIP ARTIFACT ---"
# shutil.make_archive often doubles the extension if you include it in the base_name
find . -name "*.zip*"

echo "--- 2. MANIFEST VS ACTUAL PATHS ---"
# Check what the solver thinks the output directory is
grep -r "output_directory" . | head -n 5

echo "--- 3. VERIFYING ARCHIVER SOURCE ---"
# Check if the source directory actually had files before the move
python3 -c "import os; from pathlib import Path; \
print('Staging dir check:'); \
p = Path('navier_stokes_output'); \
print(f'{p} exists: {p.exists()}'); \
if p.exists(): print(os.listdir(p))"

echo "--- 4. TEST-ARCHIVER FILENAME SYNC ---"
# Compare the filename the test expects vs the archiver hardcode
grep "final_destination =" src/common/archive_service.py
grep "zip" tests/property_integrity/test_heavy_elasticity_lifecycle.py