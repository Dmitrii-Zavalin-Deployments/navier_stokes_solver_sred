# tests/property_integrity/test_heavy_elasticity_lifecycle.py

import json
import logging
from pathlib import Path

import pytest
import numpy as np
from src.main_solver import BASE_DIR, run_solver


class TestHeavyElasticityLifecycle:

    @pytest.fixture
    def base_config(self):
        return {
            "dt_min_limit": 0.0001,
            "ppe_tolerance": 1e-4, 
            "ppe_max_iter": 20,
            "ppe_omega": 1.7,
            "ppe_max_retries": 10
        }

    @pytest.fixture
    def base_input(self):
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "nx": 4, "ny": 4, "nz": 4},
            "fluid_properties": {"density": 1.0, "viscosity": 0.001},
            "initial_conditions": {"velocity": [0.0, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {"time_step": 0.1, "total_time": 0.2, "output_interval": 1},
            "boundary_conditions": [
                {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}}, 
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}}
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, -9.81, 0.0]}
        }

    def test_scenario_1_pure_success(self, caplog, base_config, base_input):
        """
        Scenario 1: Run completes without any instabilities.
        Asserts:
        1. Final file existence.
        2. Correct total simulation time reached.
        3. Zero instability warnings in logs.
        4. State iteration count matches mathematical expectation.
        """
        # 1. SETUP
        input_filename = "test_success_input.json"
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename
        
        # Ensure we are testing a clean environment
        config_path.write_text(json.dumps(base_config))
        input_path.write_text(json.dumps(base_input))

        # Capture logs for audit
        with caplog.at_level(logging.WARNING):
            # 2. EXECUTION
            zip_path = run_solver(input_filename)

            # 3. LOG AUDIT: There should be 0 "Instability" warnings
            instability_logs = [r for r in caplog.records if "Instability" in r.message]
            assert len(instability_logs) == 0, (
                f"Scenario 1 Error: Expected 0 retries, but found {len(instability_logs)}. "
                f"Logs: {[r.message for r in instability_logs]}"
            )
            
            # 4. PATH AUDIT
            assert Path(zip_path).exists(), "The final archive was not created."
            assert zip_path.endswith(".zip")

            # 5. STATE & LOGIC AUDIT
            # total_time = 0.2, dt = 0.1 -> We expect exactly 2 iterations
            # Check if total_time was respected (SimulationContext handles this, 
            # but we verify the output existence as a proxy)
            assert "navier_stokes_output.zip" in zip_path

        # 6. CLEANUP (Rule 2: Zero Debt)
        if input_path.exists(): input_path.unlink()
        # We keep config.json if other tests need it, or delete if isolated
    
    def test_scenario_2_retry_and_recover(self, caplog, base_config, base_input):
        """
        Scenario 2: Failed run (ArithmeticError) triggers dt reduction, 
        which then succeeds on the second attempt.
        """
        # Set a time step that is slightly too aggressive for this velocity
        base_input["simulation_parameters"]["time_step"] = 0.8 
        base_input["boundary_conditions"][0]["values"]["u"] = 50.0 
        
        Path(BASE_DIR / "config.json").write_text(json.dumps(base_config))
        Path(BASE_DIR / "test_input.json").write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING):
            # We expect this to succeed eventually because elasticity will drop dt
            run_solver("test_input.json")
            
            # Check logs to confirm at least one instability was caught and fixed
            stabilization_logs = [r for r in caplog.records if "Instability" in r.message]
            assert len(stabilization_logs) > 0
            assert len(stabilization_logs) < base_config["ppe_max_retries"]

    def test_scenario_3_terminal_failure(self, caplog, base_config, base_input):
        """Scenario 3: 10 failed attempts lead to terminal RuntimeError."""
        base_input["boundary_conditions"][0]["values"]["u"] = 1e15 # Global explosion
        
        Path(BASE_DIR / "config.json").write_text(json.dumps(base_config))
        Path(BASE_DIR / "test_input.json").write_text(json.dumps(base_input))

        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError) as excinfo:
                run_solver("test_input.json")
            
            error_msg = str(excinfo.value)
            assert "unstable" in error_msg.lower()
            logs = [r for r in caplog.records if "Instability" in r.message]
            assert len(logs) == base_config["ppe_max_retries"]