# Run this from inside navier_stokes_solver_sred/
echo "--- START OF STEP 1 SOURCE ---"
for file in src/step1/*.py; do
    echo "========================================"
    echo "FILE: $file"
    echo "========================================"
    cat "$file"
    echo -e "\n"
done

echo "--- START OF TEST HELPERS ---"
for file in tests/helpers/*.py; do
    echo "========================================"
    echo "FILE: $file"
    echo "========================================"
    cat "$file"
    echo -e "\n"
done