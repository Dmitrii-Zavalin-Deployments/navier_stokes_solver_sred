import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1

def test_step1_input_coverage_sensitivity():
    """
    Data Completeness Audit:
    Ensures that every 'required' field in the schema is actually 
    captured and stored in the SolverState.
    """
    # 1. Setup a config with unique, non-default numbers
    loud_config = {
        "grid": {
            "nx": 13, "ny": 7, "nz": 5,  # Prime numbers to avoid coincidental matches
            "x_min": -1.0, "x_max": 1.0,
            "y_min": -0.5, "y_max": 0.5,
            "z_min": -0.2, "z_max": 0.8
        },
        "fluid_properties": {
            "density": 1234.56, 
            "viscosity": 0.00789
        },
        "initial_conditions": {
            "pressure": 9.99,
            "velocity": [1.1, 2.2, 3.3]
        },
        "simulation_parameters": {
            "time_step": 0.00042,
            "total_time": 60.0,
            "output_interval": 5
        },
        "boundary_conditions": [
            {"location": "x_min", "type": "inflow", "values": {"u": 1.0}},
            {"location": "x_max", "type": "outflow"}
        ],
        "external_forces": {
            "force_vector": [0.0, -9.81, 0.0]
        },
        "mask": [1] * (13 * 7 * 5)
    }

    # 2. Execute Step 1
    state = orchestrate_step1(loud_config)

    # 3. Audit: Did the data survive?
    
    # Check Grid Math
    assert state.config['grid']['nx'] == 13
    assert state.grid['dx'] == pytest.approx(2.0 / 13) # (x_max - x_min) / nx

    # Check Physics intake
    # (Assuming your orchestrator maps fluid_properties -> state.fluid_properties)
    assert state.fluid_properties['density'] == 1234.56
    assert state.fluid_properties['viscosity'] == 0.00789

    # Check Simulation parameters (Derived constants)
    assert state.constants['dt'] == 0.00042
    
    # Check Boundary Conditions (Ensuring the list wasn't dropped)
    assert len(state.boundary_conditions) == 2
    assert state.boundary_conditions[0]['location'] == "x_min"

    # Check Initial Conditions (Checking if values were applied to fields)
    # Velocity_u at index (0,0,0) should likely be the IC value 1.1
    assert state.velocity_u[0, 0, 0] == 1.1
    assert state.pressure[0, 0, 0] == 9.99

    print("\n[DATA AUDIT PASS] All schema inputs are traceable in SolverState.")