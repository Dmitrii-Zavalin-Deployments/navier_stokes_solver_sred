# tests/property_integrity/test_solver_lifecycle.py

import json
import os
import zipfile
from pathlib import Path
import shutil
import src.main_solver

from src.main_solver import run_solver
from tests.helpers.solver_input_schema_dummy import create_validated_input


class TestSolverLifecycle:
    """
    SYSTEM AUDITOR: Verifies the full pipeline flow 
    from Disk Input -> Numerical Loop -> Zip Archive.
    """

    def test_full_solver_pipeline_integrity(self, tmp_path, monkeypatch):
        # 1. Setup Isolated Environment (Rule 5)
        # We move the execution to a temp directory so we don't touch real files
        test_dir = tmp_path / "run_env"
        test_dir.mkdir()
        
        # Mock BASE_DIR in main_solver to point to our temp directory
        monkeypatch.setattr("src.main_solver.BASE_DIR", test_dir)

        # 2. Create Config (Deterministic numerical settings)
        config_file = test_dir / "config.json"
        config_dict = {
            "ppe_tolerance": 1e-4,
            "ppe_atol": 1e-6,
            "ppe_max_iter": 5,
            "ppe_omega": 1.2
        }
        config_file.write_text(json.dumps(config_dict))

        # 3. Create Input using SSoT Helper (Rule 4)
        # Use a tiny grid (3x3x3) to keep the system test fast
        input_obj = create_validated_input(nx=3, ny=3, nz=3)
        input_obj.simulation_parameters.total_time = 0.002
        input_obj.simulation_parameters.time_step = 0.001
        
        input_file = test_dir / "input_test.json"
        input_file.write_text(json.dumps(input_obj.to_dict()))

        # 4. Mock the Schema location
        # main_solver looks for 'schema/solver_input_schema.json' relative to BASE_DIR
        schema_dir = test_dir / "schema"
        schema_dir.mkdir()
        real_schema = Path(__file__).resolve().parent.parent.parent / "schema" / "solver_input_schema.json"
        (schema_dir / "solver_input_schema.json").write_text(real_schema.read_text())

        # 5. EXECUTION
        # We pass the filename string as run_solver expects
        os.chdir(test_dir)
        zip_path = run_solver("input_test.json")

        # 6. ASSERTIONS
        archive = Path(zip_path)
        assert archive.exists(), "CRITICAL: Pipeline finished but no zip archive found."
        assert archive.stat().st_size > 0, "CRITICAL: Archive is empty."
        
        # Log success for the audit trail
        print(f"\nDEBUG [Audit]: Successfully archived {archive.name}")

    def test_full_solver_pipeline_integrity(self, tmp_path, monkeypatch):
        # 1. Prepare Schema-Compliant Input Data (Rule 5: Zero-Default Policy)
        nx, ny, nz = 4, 4, 4
        input_data = {
            "domain_configuration": {
                "type": "INTERNAL"
            },
            "grid": {
                "x_min": 0.0, "x_max": 1.0,
                "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0,
                "nx": nx, "ny": ny, "nz": nz
            },
            "fluid_properties": {
                "density": 1.0,
                "viscosity": 0.01
            },
            "initial_conditions": {
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0
            },
            "simulation_parameters": {
                "time_step": 0.001,
                "total_time": 0.01,
                "output_interval": 1
            },
            "boundary_conditions": [
                {
                    "location": "x_min",
                    "type": "inflow",
                    "values": {"u": 1.0, "v": 0.0, "w": 0.0}
                },
                {
                    "location": "x_max",
                    "type": "outflow",
                    "values": {"p": 0.0}
                }
            ],
            # Canonical flattening: length must be nx*ny*nz (Rule 9)
            "mask": [0] * (nx * ny * nz),
            "external_forces": {
                "force_vector": [0.0, -9.81, 0.0]
            }
        }

        # 2. Setup Filesystem in tmp_path
        input_file = tmp_path / "input_test.json"
        input_file.write_text(json.dumps(input_data))

        # 3. Redirect BASE_DIR to tmp_path (Critical Alignment)
        # This ensures main_solver uses tmp_path as the root for this test
        monkeypatch.setattr(src.main_solver, "BASE_DIR", tmp_path)

        # 4. Copy the production config.json to the temp root
        # We resolve the path relative to this test file's location
        current_file_path = Path(__file__).resolve()
        # Path logic: tests/property_integrity/test_file.py -> root is 3 levels up
        project_root = current_file_path.parent.parent.parent
        real_config = project_root / "config.json"
        
        if not real_config.exists():
            raise FileNotFoundError(f"Test setup error: Could not find config.json at {real_config}")
            
        shutil.copy(real_config, tmp_path / "config.json")

        # 5. Execute Pipeline
        # run_solver will now look for BASE_DIR/input_test.json and BASE_DIR/config.json
        zip_path = run_solver("input_test.json")
        
        # 6. Verification
        final_zip = Path(zip_path)
        assert final_zip.exists(), f"Pipeline failed to produce archive at {zip_path}"
        assert final_zip.suffix == ".zip"