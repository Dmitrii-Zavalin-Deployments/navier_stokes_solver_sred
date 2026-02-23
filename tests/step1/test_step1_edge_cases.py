# tests/step1/test_step1_edge_cases.py

import pytest
import numpy as np
from src.step1.parse_config import parse_config
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.allocate_fields import allocate_fields
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_parse_config_minimal_parameters():
    """Verifies parse_config handles stripped optional keys without crashing."""
    config = solver_input_schema_dummy()
    
    # We strip 'output_interval' to check if the parser survives.
    # Since your code doesn't inject it back, we just verify the block remains valid.
    if "output_interval" in config["simulation_parameters"]:
        del config["simulation_parameters"]["output_interval"]
    
    parsed = parse_config(config)
    assert "simulation_parameters" in parsed
    assert "time_step" in parsed["simulation_parameters"]

def test_apply_initial_conditions_broadcasting():
    """
    Directly targets Missing Lines 24-25, 32 in apply_initial_conditions.py.
    Forces the code to broadcast scalars/lists to the 3D staggered grid.
    """
    nx, ny, nz = 2, 2, 2
    fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz))
    }
    
    # Passing a list for velocity and a scalar for pressure to trigger broadcasting logic
    ic = {
        "velocity": [1.0, 0.0, 0.0],
        "pressure": 5.0
    }
    
    apply_initial_conditions(fields, ic)
    
    # TRUTH: Staggered U-velocity should be filled with 1.0
    assert np.all(fields["U"] == 1.0)
    # TRUTH: Cell-centered Pressure should be filled with 5.0
    assert np.all(fields["P"] == 5.0)

def test_compute_derived_constants_physics_coverage():
    """
    Directly targets Missing Lines 36, 39 in compute_derived_constants.py.
    Provides manual spatial steps to ensure the translator logic is executed.
    """
    dummy = solver_input_schema_dummy()
    grid_data = dummy["grid"]
    
    # Manually inject dx, dy, dz which are often missing in raw config but 
    # required for derived physics calculations.
    grid_data.update({"dx": 0.1, "dy": 0.1, "dz": 0.1})
    
    constants = compute_derived_constants(
        grid_data, 
        dummy["fluid_properties"], 
        dummy["simulation_parameters"]
    )
    
    assert "rho" in constants
    assert "dt" in constants
    assert constants["rho"] == 1.0

def test_allocate_fields_staggered_audit():
    """Ensures 100% coverage for allocate_fields.py by verifying memory layout."""
    config = solver_input_schema_dummy()
    fields = allocate_fields(config["grid"])
    
    # Audit the Arakawa C-grid shapes (N+1 for faces)
    assert fields["U"].shape == (3, 2, 2)
    assert fields["V"].shape == (2, 3, 2)
    assert fields["W"].shape == (2, 2, 3)
    assert fields["P"].shape == (2, 2, 2)