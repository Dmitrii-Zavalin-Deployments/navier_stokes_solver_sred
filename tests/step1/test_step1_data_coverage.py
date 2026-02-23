# tests/step1/test_step1_data_coverage.py

import pytest
import numpy as np
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_step1_input_coverage_sensitivity():
    """
    Data Completeness Audit (Phase E):
    Uses the Canonical Dummy as a base and applies 'Loud' overrides 
    to verify data transport from JSON to SolverState.
    """
    # 1. Get the base dummy that satisfies the schema
    input_data = solver_input_schema_dummy()

    # 2. Inject "Loud" overrides (Unique values to ensure they aren't defaults)
    input_data["grid"]["nx"] = 13
    input_data["grid"]["ny"] = 7
    input_data["grid"]["nz"] = 5
    input_data["grid"]["x_min"] = -2.5
    input_data["grid"]["x_max"] = 2.5  # Length = 5.0
    
    input_data["fluid_properties"]["density"] = 1234.56
    input_data["fluid_properties"]["viscosity"] = 0.00789
    
    input_data["simulation_parameters"]["time_step"] = 0.00042
    
    input_data["initial_conditions"]["pressure"] = 9.99
    input_data["initial_conditions"]["velocity"] = [1.1, 2.2, 3.3]

    # Re-generate mask to match the new 'loud' dimensions
    total_cells = 13 * 7 * 5
    input_data["mask"] = [1] * total_cells

    # 3. Execute Step 1 Pipeline
    state = orchestrate_step1(input_data)

    # 4. Audit: Assert that the 'Loud' values survived the intake
    
    # Grid Math Sensitivity
    assert state.config['grid']['nx'] == 13
    # dx = (2.5 - (-2.5)) / 13 = 5.0 / 13
    assert state.grid['dx'] == pytest.approx(5.0 / 13)

    # Physics Property Sensitivity
    assert state.fluid_properties['density'] == 1234.56
    assert state.fluid_properties['viscosity'] == 0.00789

    # Derived Constant Sensitivity
    assert state.constants['dt'] == 0.00042
    
    # Initial Condition Field Application
    # We check if the 'loud' ICs were actually written into the numpy arrays
    assert state.velocity_u[0, 0, 0] == 1.1
    assert state.velocity_v[0, 0, 0] == 2.2
    assert state.velocity_w[0, 0, 0] == 3.3
    assert state.pressure[0, 0, 0] == 9.99

    # Boundary Condition Structure Survival
    assert len(state.boundary_conditions) == 6
    assert "x_min" in state.boundary_conditions

    print("\n[DATA AUDIT PASS] Canonical Dummy data successfully traced to SolverState.")