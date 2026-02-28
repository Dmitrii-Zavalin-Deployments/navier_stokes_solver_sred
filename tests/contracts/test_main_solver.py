# tests/contracts/test_main_solver.py

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.main_solver import run_solver_from_file

# Import your helpers
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy # Assuming this is the step 5 final output

class TestMainSolverOrchestration:

    @patch("src.main_solver.orchestrate_step1")
    @patch("src.main_solver.orchestrate_step2")
    @patch("src.main_solver.orchestrate_step3")
    @patch("src.main_solver.orchestrate_step4")
    @patch("src.main_solver.orchestrate_step5")
    @patch("src.main_solver.validate_final_state")
    @patch("src.main_solver.archive_simulation_artifacts")
    def test_main_solver_full_flow(
        self, mock_arch, mock_val, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1, tmp_path
    ):
        """
        Tests the integration of main_solver using dummy stubs.
        This verifies that each step receives the output of the previous one.
        """
        # 1. Setup Input
        input_file = tmp_path / "test_input.json"
        input_file.write_text(json.dumps(solver_input_schema_dummy()))
        
        # 2. Setup Dummy Returns (The Stubbing)
        mock_s1.return_value = make_step1_output_dummy()
        mock_s2.return_value = make_step2_output_dummy()
        mock_s3.return_value = make_step3_output_dummy()
        mock_s4.return_value = make_step4_output_dummy()
        mock_s5.return_value = make_output_schema_dummy()
        
        mock_arch.return_value = "simulation_results.zip"

        # 3. Execute
        result = run_solver_from_file(str(input_file))
        
        # 4. Assertions
        assert result == "simulation_results.zip"
        
        # Verify the chain of custody
        mock_s1.assert_called_once()
        mock_s2.assert_called_with(mock_s1.return_value)
        mock_s3.assert_called_with(mock_s2.return_value, current_time=0.0, step_index=0)
        mock_s4.assert_called_with(mock_s3.return_value)
        mock_s5.assert_called_with(mock_s4.return_value)
        
        mock_val.assert_called_once_with(mock_s5.return_value)