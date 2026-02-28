# tests/contracts/test_main_solver.py

import os
import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.main_solver import run_solver_from_file, archive_simulation_artifacts

# Import your helpers
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

class TestMainSolverOrchestration:

    # --- HAPPY PATH ---
    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.orchestrate_step2")
    @patch("src.main_solver.orchestrate_step3")
    @patch("src.main_solver.orchestrate_step4")
    @patch("src.main_solver.orchestrate_step5")
    @patch("src.main_solver.archive_simulation_artifacts")
    def test_main_solver_full_flow_success(
        self, mock_arch, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1, tmp_path
    ):
        """Tests successful pipeline integration and chain of custody."""
        input_file = tmp_path / "test_input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        mock_s1.return_value = make_step1_output_dummy()
        mock_s2.return_value = make_step2_output_dummy()
        mock_s3.return_value = make_step3_output_dummy()
        mock_s4.return_value = make_step4_output_dummy()
        mock_s5.return_value = make_output_schema_dummy()
        mock_arch.return_value = "simulation_results.zip"

        result = run_solver_from_file(str(input_file))
        
        assert result == "simulation_results.zip"
        mock_s3.assert_called_with(mock_s2.return_value, current_time=0.0, step_index=0)
        mock_arch.assert_called_once_with(mock_s5.return_value)

    # --- EDGE CASE: INPUT ERRORS ---
    def test_run_solver_file_not_found(self):
        """Ensures FileNotFoundError is raised for missing input."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            run_solver_from_file("non_existent_path.json")

    def test_run_solver_malformed_json(self, tmp_path):
        """Ensures ValueError is raised for invalid JSON syntax."""
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{ 'invalid': json }")
        with pytest.raises(ValueError, match="Failed to parse input JSON"):
            run_solver_from_file(str(bad_file))

    # --- EDGE CASE: PIPELINE CRASH ---
    @patch("src.main_solver.orchestrate_step1")
    def test_run_solver_pipeline_failure(self, mock_s1, tmp_path):
        """Ensures RuntimeError is raised if a step crashes."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        mock_s1.side_effect = Exception("Matrix Solver Diverged")
        
        with pytest.raises(RuntimeError, match="Solver Pipeline crashed"):
            run_solver_from_file(str(input_file))

    # --- ARCHIVE LOGIC & MISSING FILES ---
    @patch("shutil.make_archive")
    def test_archive_artifacts_missing_snapshots(self, mock_zip, tmp_path):
        """Tests archiver's resilience when snapshots listed in manifest don't exist."""
        state = make_output_schema_dummy()
        # Simulate a manifest pointing to a file that doesn't exist
        state.manifest.saved_snapshots = ["/tmp/ghost_file.vtk"]
        
        # We need to mock Path.exists to return False for the snapshot, but True for dirs
        with patch.object(Path, "exists", side_effect=lambda self: "navier-stokes-output" in str(self) or "data" in str(self)):
             result = archive_simulation_artifacts(state)
             # Should still return a path to the zip even if snapshot copying was skipped
             assert "navier-stokes-output.zip" in result

    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.archive_simulation_artifacts")
    def test_run_solver_archive_failure(self, mock_arch, mock_s1, tmp_path):
        """Ensures RuntimeError is raised if zipping fails after a successful run."""
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        mock_s1.return_value = make_step1_output_dummy()
        mock_arch.side_effect = Exception("Disk Full")
        
        # Mock other steps to just pass the state through
        with patch("src.main_solver.orchestrate_step2"), \
             patch("src.main_solver.orchestrate_step3"), \
             patch("src.main_solver.orchestrate_step4"), \
             patch("src.main_solver.orchestrate_step5"):
            
            with pytest.raises(RuntimeError, match="ARCHIVE FAILURE"):
                run_solver_from_file(str(input_file))