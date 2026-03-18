# tests/property_integrity/test_solver_lifecycle.py

import json
from pathlib import Path

from src.main_solver import BASE_DIR, run_solver


class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: High-Fidelity Smoke Test.
    Uses the EXACT production filename 'input_test.json'.
    Ensures the 'Plumbing' is ready for the real Dropbox download.
    """

    def test_production_plumbing_gatekeeper(self):
        # 1. Target: The exact production filename
        test_filename = "input_test.json"
        input_path = Path(BASE_DIR) / test_filename
        
        # 2. Payload: Minimal 4x4x4 grid (Schema-compliant)
        # Keeping it small ensures sub-second execution while testing all steps.
        nx, ny, nz = 4, 4, 4
        input_data = {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0, "x_max": 1.0,
                "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0,
                "nx": nx, "ny": ny, "nz": nz
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.01},
            "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 0.0},
            "simulation_parameters": {
                "time_step": 0.001,
                "total_time": 0.002, 
                "output_interval": 1
            },
            "boundary_conditions": [
                {
                    "location": "x_min", "type": "inflow",
                    "values": {"u": 1.0, "v": 0.0, "w": 0.0}
                },
                {
                    "location": "x_max", "type": "outflow",
                    "values": {"p": 0.0}
                }
            ],
            # Canonical flattening: length must be nx*ny*nz
            "mask": [0] * (nx * ny * nz),
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

        zip_path = None
        output_dir = Path(BASE_DIR) / "data" / "testing-input-output"

        try:
            # 3. Inject: Create the file in the repo root
            input_path.write_text(json.dumps(input_data))

            # 4. Execute: Call the real production entry point
            # This validates: Root Access -> Schema Load -> Step 1-5 -> Zip Creation
            zip_path_str = run_solver(test_filename)
            zip_path = Path(zip_path_str)

            # 5. Assert: Verify the plumbing success
            assert zip_path.exists(), "GATEKEEPER FAIL: ZIP not created."
            assert zip_path.parent == output_dir, f"GATEKEEPER FAIL: Wrong output dir. Expected {output_dir}"
            assert zip_path.stat().st_size > 0, "GATEKEEPER FAIL: ZIP is empty."

        finally:
            # 6. PURGE: Wipe everything to leave an empty folder for Dropbox download
            if input_path.exists():
                input_path.unlink()
            
            if output_dir.exists():
                # Remove ANY .zip starting with 'input_test' to ensure no confusion with prod data
                for artifact in output_dir.glob("input_test*.zip"):
                    artifact.unlink()
                    print(f"\n[Sanitization] Purged smoke test artifact: {artifact.name}")
            
            # 7. FINAL AUDIT: Assert the folder is 100% clean
            # This ensures no "ghost" results remain before the real simulation starts.
            leftover_zips = list(output_dir.glob("*.zip"))
            assert len(leftover_zips) == 0, f"CLEANUP FAILURE: Found {len(leftover_zips)} leftover artifacts."