# tests/test_main_solver.py

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
        # Check that Step 1 was called with a SolverInput object, not a dict
        args, _ = mock_s1.call_args
        from src.solver_input import SolverInput
        assert isinstance(args[0], SolverInput)

    def test_run_solver_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_solver_from_file("non_existent.json")

    def test_run_solver_malformed_json(self, tmp_path):
        bad_file = tmp_path / "broken.json"
        bad_file.write_text("{ invalid }")
        with pytest.raises(ValueError, match="Failed to parse input JSON"):
            run_solver_from_file(str(bad_file))

    def test_run_solver_triage_failure(self, tmp_path):
        """Tests that bad physics (negative density) is caught by the Triage Gate."""
        bad_input = solver_input_schema_dummy()
        bad_input["fluid_properties"]["density"] = -5.0
        
        input_file = tmp_path / "bad_physics.json"
        input_file.write_text(json.dumps(bad_input))
        
        with pytest.raises(ValueError, match="Input data failed triage"):
            run_solver_from_file(str(input_file))

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
        
        def exists_side_effect(*args, **kwargs):
            path_str = str(args[0]) if args else ""
            return "navier-stokes-output" in path_str or "data" in path_str

        with patch.object(Path, "exists", side_effect=exists_side_effect):
            with patch.object(Path, "mkdir"):
                with patch("builtins.open", MagicMock()):
                    result = archive_simulation_artifacts(state)
                    mock_zip.assert_called_once()
                    
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

    @patch("builtins.open")
    def test_run_solver_generic_read_error(self, mock_open, tmp_path):
        input_file = tmp_path / "perm_error.json"
        input_file.write_text("{}") 
        mock_open.side_effect = RuntimeError("Permission Denied")
        with pytest.raises(RuntimeError, match="Unexpected error reading input file"):
            run_solver_from_file(str(input_file))

    @patch("shutil.make_archive")
    @patch("shutil.rmtree")
    @patch("pathlib.Path.exists")
    def test_archive_cleanup_existing_dir(self, mock_exists, mock_rmtree, mock_zip):
        state = make_output_schema_dummy()
        state.manifest.saved_snapshots = []
        mock_exists.side_effect = [True, False, False]
        
        with patch("pathlib.Path.mkdir"), patch("builtins.open", MagicMock()):
            archive_simulation_artifacts(state)
        assert mock_rmtree.called

    @patch("shutil.make_archive")
    @patch("shutil.copy2")
    def test_archive_successful_snapshot_copy(self, mock_copy, mock_zip, tmp_path):
        state = make_output_schema_dummy()
        fake_snapshot = tmp_path / "real_snapshot.vtk"
        fake_snapshot.write_text("data")
        state.manifest.saved_snapshots = [str(fake_snapshot)]
        with patch("pathlib.Path.mkdir"), patch("builtins.open", MagicMock()):
            archive_simulation_artifacts(state)
            mock_copy.assert_called_once()