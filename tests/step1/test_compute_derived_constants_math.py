# tests/step1/test_compute_derived_constants_math.py

import pytest
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_compute_derived_constants_matches_dummy():
    """
    Verifies that Step 1 correctly extracts and calculates fundamental 
    constants (rho, mu, dt, dx, dy, dz) into the SolverState object.
    """

    # 1. Start with the canonical, schema-valid dummy
    json_input = solver_input_schema_dummy()

    # 2. Override values to test specific math logic
    # We set (max - min) / n = 1.0 to verify unit spacing calculation
    json_input["domain"].update({
        "nx": 2, "x_min": 0.0, "x_max": 2.0,
        "ny": 2, "y_min": 0.0, "y_max": 2.0,
        "nz": 2, "z_min": 0.0, "z_max": 2.0
    })
    
    json_input["fluid_properties"] = {
        "density": 10.0,
        "viscosity": 2.0,
    }
    json_input["simulation_parameters"]["time_step"] = 0.1

    # 3. Run Orchestrator (Returns SolverState object)
    state = orchestrate_step1_state(json_input)

    # 4. Assertions using Object Attribute Access (.constants)
    # The internal 'constants' is still a dict, so we use ['key'] for the second level
    assert state.constants["rho"] == 10.0
    assert state.constants["mu"] == 2.0
    assert state.constants["dt"] == 0.1

    # Verify grid spacing calculations: (2.0 - 0.0) / 2 = 1.0
    assert state.constants["dx"] == 1.0
    assert state.constants["dy"] == 1.0
    assert state.constants["dz"] == 1.0

    # Extra check: Verify these constants are also reflected in the grid metadata
    assert state.grid["dx"] == 1.0