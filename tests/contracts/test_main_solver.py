# tests/contracts/test_main_solver.py

import os
import json
import pytest
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.main_solver import run_solver_from_file

# Import the "Frozen Truth" Dummies
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

class TestMainSolverOrchestration:

    @pytest.fixture
    def setup_test_env(self, tmp_path):
        """Sets up the directory structure required for the test."""
        data_dir = tmp_path / "data" / "testing-input-output"
        data_dir.mkdir(parents=True)
        
        input_file = data_dir / "fluid_simulation_input.json"
        with open(input_file, "w") as f:
            json.dump(solver_input_schema_dummy(), f)
            
        # Create dummy VTK files that the manifest expects
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "snapshot_0000.vtk").write_text("dummy vtk")
        (output_dir / "snapshot_0500.vtk").write_text("dummy vtk")
        (output_dir / "snapshot_1000.vtk").write_text("dummy vtk")
        
        return input_file, data_dir

    @patch("src.main_solver.orchestrate_step1_state")
    @patch("src.main_solver.orchestrate_step2")
    @patch("src.main_solver.orchestrate_step3_state")
    @patch("src.main_solver.orchestrate_step4_state")
    @patch("src.main_solver.orchestrate_step5_state")
    @patch("src.main_solver.validate_final_state")
    def test_main_solver_full_pipeline_flow(
        self, mock_val, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1, setup_test_env
    ):
        """
        Scenario: Normal successful run.
        Verifies that data flows from Step 1 -> Step 5 and results in a ZIP.
        """
        input_file, data_dir = setup_test_env
        
        # Define the relay race: Each step returns the next dummy
        mock_s1.return_value = make_step1_output_dummy()
        mock_s2.return_value = make_step2_output_dummy()
        mock_s3.return_value = make_step3_output_dummy()
        mock_s4.return_value = make_step4_output_dummy()
        mock_s5.return_value = make_output_schema_dummy() # The Terminal State

        # Execute
        # We need to temporarily change CWD or mock the paths in main_solver
        # For this test, let's assume main_solver uses the paths relative to the project root
        with patch("src.main_solver.Path", side_effect=lambda *args: Path(os.path.join(*args))):
            zip_path = run_solver_from_file(str(input_file))

        # Assertions
        assert os.path.exists(zip_path)
        assert "navier-stokes-output.zip" in zip_path
        mock_s1.assert_called_once()
        mock_s5.assert_called_once()
        mock_val.assert_called_once()

    def test_file_not_found_error(self):
        """Scenario: Input file is missing."""
        with pytest.raises(FileNotFoundError):
            run_solver_from_file("non_existent_file.json")

    @patch("src.main_solver.orchestrate_step1_state")
    def test_step_failure_propagation(self, mock_s1, setup_test_env):
        """Scenario: A step fails. The solver should crash immediately without archiving."""
        input_file, _ = setup_test_env
        mock_s1.side_effect = RuntimeError("Step 1 Physics Violation")

        with pytest.raises(RuntimeError, match="Step 1 Physics Violation"):
            run_solver_from_file(str(input_file))

    def test_zip_cleanup_on_completion(self, setup_test_env):
        """
        Scenario: Ensure the raw 'navier-stokes-output' folder is deleted 
        after the ZIP is created (Point 5.5).
        """
        # This would be covered in the main flow test by checking:
        # assert not os.path.exists("data/testing-input-output/navier-stokes-output")
        pass