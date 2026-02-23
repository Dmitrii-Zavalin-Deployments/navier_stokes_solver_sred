# tests/step1/test_compute_derived_constants_math.py

import pytest
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_compute_derived_constants_mathematical_precision():
    """
    Surveyor Audit: Verifies spatial (dx, dy, dz) and physical (rho, mu, dt) 
    constant derivation from raw JSON.
    """

    # 1. Load canonical dummy (defaults to 2x2x2)
    json_input = solver_input_schema_dummy()

    # 2. Inject specific values for precision verification
    nx, ny, nz = 10, 5, 2
    json_input["grid"].update({
        "nx": nx, "x_min": -5.0, "x_max": 5.0, # dx = 10/10 = 1.0
        "ny": ny, "y_min": 0.0,  "y_max": 5.0, # dy = 5/5   = 1.0
        "nz": nz, "z_min": 0.0,  "z_max": 1.0  # dz = 1/2   = 0.5
    })
    
    # CRITICAL FIX: Update mask length to match new grid (10 * 5 * 2 = 100)
    json_input["mask"] = [0] * (nx * ny * nz)
    
    json_input["fluid_properties"] = {
        "density": 997.0,
        "viscosity": 0.001,
    }
    json_input["simulation_parameters"]["time_step"] = 0.005

    # 3. Execute Step 1 Orchestrator
    state = orchestrate_step1_state(json_input)

    # 4. Assert Physical Constants
    assert state.constants["rho"] == 997.0
    assert state.constants["mu"] == 0.001
    assert state.constants["dt"] == 0.005

    # 5. Assert Spatial Metrics (Delta Calculations)
    assert state.constants["dx"] == 1.0
    assert state.constants["dy"] == 1.0
    assert state.constants["dz"] == 0.5

    # 6. Metadata Synchronization
    assert state.grid["dx"] == state.constants["dx"]
    assert state.grid["dy"] == state.constants["dy"]
    assert state.grid["dz"] == state.constants["dz"]