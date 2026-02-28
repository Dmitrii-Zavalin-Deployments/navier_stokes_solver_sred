import os
import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.main_solver import run_solver_from_file, archive_simulation_artifacts

from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

class TestMainSolverOrchestration:

    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.orchestrate_step2")
    @patch("src.main_solver.orchestrate_step3")
    @patch("src.main_solver.orchestrate_step4")
    @patch("src.main_solver.orchestrate_step5")
    # Note: We do NOT mock archive_simulation_artifacts here to get coverage on it
    @patch("shutil.make_archive") 
    @patch("shutil.rmtree")
    def test_main_solver_full_flow_success(
        self, mock_rm, mock_zip, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1, tmp_path
    ):
        input_file = tmp_path / "test_input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        mock_s1.return_value = make_step1_output_dummy()
        mock_s2.return_value = make_step2_output_dummy()
        mock_s3.return_value = make_step3_output_dummy()
        mock_s4.return_value = make_step4_output_dummy()
        mock_s5.return_value = make_output_schema_dummy()
        mock_zip.return_value = "simulation_results.zip"

        result = run_solver_from_file(str(input_file))
        assert "simulation_results.zip" in result

    def test_run_solver_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_solver_from_file("non_existent.json")

    def test_run_solver_malformed_json(self, tmp_path):
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{ invalid }")
        with pytest.raises(ValueError):
            run_solver_from_file(str(bad_file))

    @patch("src.main_solver.orchestrate_step1")
    def test_run_solver_pipeline_failure(self, mock_s1, tmp_path):
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        mock_s1.side_effect = Exception("Crash")
        with pytest.raises(RuntimeError, match="Solver Pipeline crashed"):
            run_solver_from_file(str(input_file))

    @patch("shutil.make_archive")
    def test_archive_artifacts_missing_snapshots(self, mock_zip, tmp_path):
        state = make_output_schema_dummy()
        state.manifest.saved_snapshots = ["ghost.vtk"]
        
        # The lambda fix: accept 'self' (the Path instance)
        with patch.object(Path, "exists", side_effect=lambda p: "navier-stokes-output" in str(p)):
            with patch.object(Path, "mkdir"):
                # Mock open to avoid actual file writes
                with patch("builtins.open", MagicMock()):
                    result = archive_simulation_artifacts(state)
                    assert "navier-stokes-output" in result

    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.archive_simulation_artifacts")
    def test_run_solver_archive_failure(self, mock_arch, mock_s1, tmp_path):
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        mock_s1.return_value = make_step1_output_dummy()
        mock_arch.side_effect = Exception("IO Error")
        
        with patch("src.main_solver.orchestrate_step2"), \
             patch("src.main_solver.orchestrate_step3"), \
             patch("src.main_solver.orchestrate_step4"), \
             patch("src.main_solver.orchestrate_step5"):
            with pytest.raises(RuntimeError, match="ARCHIVE FAILURE"):
                run_solver_from_file(str(input_file))