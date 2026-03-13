# tests/integration/test_solver_lifecycle.py

import os
import json
from pathlib import Path

from src.main_solver import run_solver


class TestSolverLifecycle:
    """
    INTEGRATION AUDITOR: Verifies the full pipeline flow 
    from Initialization (Step 1) to Archive (Step 5).
    """

    def test_full_solver_pipeline_integrity(self, tmp_path):
        """
        Rule 7 (STS): Atomic Verification of the end-to-end 
        solver lifecycle using a temporary filesystem sandbox.
        """
        # 1. Define paths relative to the project root (where the solver looks)
        # We assume the solver is being run in an environment where it can find the root.
        # For testing, we mock the inputs in the project root's structure.
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # 2. Mocking the required files in a location the solver can access
        # Since run_solver uses BASE_DIR, we ensure these exist at project root.
        config_file = project_root / "config.json"
        input_file = project_root / "input_validated.json"
        
        # Save original state to restore after test
        orig_config = config_file.read_text() if config_file.exists() else None
        
        try:
            # Write test mocks
            config_file.write_text(json.dumps({
                "ppe_tolerance": 1e-6, 
                "ppe_max_iter": 10, 
                "ppe_omega": 1.0
            }))
            input_file.write_text(json.dumps({
                "simulation_parameters": {
                    "time_step": 0.01, 
                    "total_time": 0.02, 
                    "output_interval": 1
                }, 
                "domain": {"case_name": "test_case", "type": "INTERNAL"}
            }))
            
            # 3. Execution
            final_state = run_solver("input_validated.json")
            
            # 4. Assertions
            assert final_state.ready_for_time_loop is False
            assert len(final_state.manifest.saved_snapshots) > 0
            
            # 5. Archive Verification
            archive_path = project_root / "navier_stokes_test_case_output.zip"
            assert archive_path.exists(), "Archive service failed to produce output."
            
            # Cleanup
            archive_path.unlink()
            
        finally:
            # Restore original config if it existed
            if orig_config:
                config_file.write_text(orig_config)
            input_file.unlink()