# tests/step1/test_compute_derived_constants.py

import pytest
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_step1_constants_match_dummy():
    """
    Verifies that Step 1 orchestrator correctly populates the constants 
    attribute of the SolverState object from the JSON configuration.
    Follows Rule: Inherit from Dummy, reassign for specific verification.
    """

    # 1. Start with the canonical, schema-valid dummy (Constitutional Alignment)
    json_input = solver_input_schema_dummy()

    # 2. Reassign specific values to verify the math logic
    # Setting (max - min) / n = 1.0 to check spacing logic explicitly
    json_input["grid"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_min": 0.0, "x_max": 2.0,
        "y_min": 0.0, "y_max": 2.0,
        "z_min": 0.0, "z_max": 2.0
    })

    json_input["fluid_properties"] = {
        "density": 5.0,
        "viscosity": 0.2,
    }

    json_input["simulation_parameters"].update({
        "time_step": 0.05,
        "total_time": 1.0,
        "output_interval": 1
    })

    # 3. Execute Step 1 (Returns SolverState object)
    state = orchestrate_step1_state(json_input)

    # 4. Assertions for Fluid Properties
    assert state.constants["rho"] == 5.0
    assert state.constants["mu"] == 0.2
    assert state.constants["dt"] == 0.05

    # 5. Verify spacing calculation: (2.0 - 0.0) / 2 = 1.0
    # This proves the orchestrator is correctly deriving physical units from coordinates
    assert state.constants["dx"] == 1.0
    assert state.constants["dy"] == 1.0
    assert state.constants["dz"] == 1.0

    # Ensure "Pure Path" access: constants must mirror the grid metadata
    assert state.grid["dx"] == 1.0
    assert state.grid["dy"] == 1.0
    assert state.grid["dz"] == 1.0