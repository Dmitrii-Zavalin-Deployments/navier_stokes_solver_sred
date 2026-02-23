# tests/step1/test_compute_derived_constants_math.py

import pytest
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_compute_derived_constants_mathematical_precision():
    """
    Surveyor Audit: Verifies spatial (dx, dy, dz) and physical (rho, mu, dt) 
    constant derivation from raw JSON.
    """

    # 1. Load canonical dummy
    json_input = solver_input_schema_dummy()

    # 2. Inject specific values for precision verification
    # dx = (5.0 - (-5.0)) / 10 = 1.0
    json_input["grid"].update({
        "nx": 10, "x_min": -5.0, "x_max": 5.0,
        "ny": 5,  "y_min": 0.0,  "y_max": 5.0,   # dy = 1.0
        "nz": 2,  "z_min": 0.0,  "z_max": 1.0    # dz = 0.5
    })
    
    json_input["fluid_properties"] = {
        "density": 997.0,
        "viscosity": 0.001,
    }
    json_input["simulation_parameters"]["time_step"] = 0.005

    # 3. Execute Step 1 Orchestrator
    state = orchestrate_step1_state(json_input)

    # 4. Assert Physical Constants (Internal Shorthand)
    assert state.constants["rho"] == 997.0, "Density mapping failed."
    assert state.constants["mu"] == 0.001, "Viscosity mapping failed."
    assert state.constants["dt"] == 0.005, "Time-step mapping failed."

    # 5. Assert Spatial Metrics (Delta Calculations)
    # Calculation: (Max - Min) / N_cells
    assert state.constants["dx"] == 1.0
    assert state.constants["dy"] == 1.0
    assert state.constants["dz"] == 0.5

    # 6. Metadata Synchronization
    # Ensures grid metadata and internal constants are identical
    assert state.grid["dx"] == state.constants["dx"]
    assert state.grid["dy"] == state.constants["dy"]
    assert state.grid["dz"] == state.constants["dz"]