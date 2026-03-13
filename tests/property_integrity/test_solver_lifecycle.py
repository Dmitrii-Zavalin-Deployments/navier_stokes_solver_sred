# tests/integration/test_solver_lifecycle.py

import pytest
import os
from pathlib import Path
from src.main_solver import run_solver
from src.common.solver_state import SolverState

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
        # 1. Setup temporary sandbox for this test iteration
        os.chdir(tmp_path)
        os.mkdir("schema")
        
        # 2. Mocking the required files for the main_solver to function
        # Per Rule 5 (Deterministic Init), these must be present.
        with open("config.json", "w") as f:
            f.write('{"ppe_tolerance": 1e-6, "ppe_max_iter": 10, "ppe_omega": 1.0}')
        
        # 3. Execution: Orchestrate the full lifecycle
        # In a real environment, you'd point to your 'input_validated.json'
        # We ensure that run_solver returns a valid terminal state.
        final_state = run_solver("input_validated.json")
        
        # 4. Assertions: Verifying the State Hierarchy (Rule 4)
        assert final_state.ready_for_time_loop is False, "Solver must exit cleanly."
        assert len(final_state.manifest.saved_snapshots) > 0, "Archivist must have registered snapshots."
        
        # 5. Archive Verification
        # Check that the archive service actually created the zip file
        archive_path = Path("navier_stokes_default_output.zip")
        assert archive_path.exists(), "Archive service failed to produce output."