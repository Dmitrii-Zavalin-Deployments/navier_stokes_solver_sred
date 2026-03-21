echo "===================================================="
echo "          NAV-STOKES CI: FORENSIC DATA DUMP          "
echo "===================================================="

echo "--- 1. WORKSPACE HIERARCHY ---"
# Shows the actual directory structure after the crash
find . -maxdepth 3 -not -path '*/.*'

echo -e "\n--- 2. CONFIGURATION VALIDATION ---"
[ -f "config.json" ] && cat config.json || echo "MISSING: config.json"

echo -e "\n--- 3. ZIP ARCHIVE TRACE ---"
# Search for any .zip files generated during the panic
find . -name "*.zip" -exec ls -lh {} +

echo -e "\n--- 4. LOG FILTRATION (PANIC EVENTS) ---"
# If you have a persistent log file, extract the panic transitions
grep -i "PANIC" elasticity.log || echo "No PANIC entries found in elasticity.log"

echo -e "\n--- 5. PERMISSIONS AUDIT ---"
# Check if the runner has write access to the output/ directory
ls -ld output/ || echo "DIRECTORY MISSING: output/"

echo "===================================================="
echo "          AUDIT COMPLETE: UPLOAD ARTIFACTS           "
echo "===================================================="