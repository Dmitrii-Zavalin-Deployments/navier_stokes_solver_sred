# tests/step1/test_step1_edge_cases.py

import pytest
import numpy as np
from src.step1.parse_config import parse_config
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.allocate_fields import allocate_fields
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_parse_config_defaults_via_inheritance():
    """Triggers fallback logic for output_interval by providing only time_step."""
    config = solver_input_schema_dummy()
    
    # To hit the 'default' line for output_interval, we must keep the dict
    # but remove the specific key. If we delete the whole dict, we hit a different branch.
    if "output_interval" in config["simulation_parameters"]:
        del config["simulation_parameters"]["output_interval"]
    
    parsed = parse_config(config)
    
    # Verify the fallback logic (Line 31 approx) injected the default
    assert "output_interval" in parsed["simulation_parameters"]
    assert parsed["simulation_parameters"]["output_interval"] == 1

def test_apply_initial_conditions_broadcasting():
    """Triggers array broadcasting for velocity components and pressure."""
    nx, ny, nz = 2, 2, 2
    fields = {
        "U": np.zeros((nx+1, ny, nz)),
        "V": np.zeros((nx, ny+1, nz)),
        "W": np.zeros((nx, ny, nz+1)),
        "P": np.zeros((nx, ny, nz))
    }
    
    # Test broadcasting a simple scalar for pressure and list for velocity
    ic = {
        "velocity": [1.0, 2.0, 3.0],
        "pressure": 0.5
    }
    
    apply_initial_conditions(fields, ic)
    
    # Verify BROADCASTING (Logic coverage for lines 24-32)
    assert np.all(fields["U"] == 1.0)
    assert np.all(fields["P"] == 0.5)

def test_compute_derived_constants_minimal():
    """Triggers physics derivations by satisfying mandatory spatial keys."""
    dummy = solver_input_schema_dummy()
    
    # The function failed because it expected 'dx' inside the grid dict.
    # We provide the grid sub-dict directly from the dummy to ensure keys exist.
    grid_data = dummy["grid"]
    # Ensure dx, dy, dz are present (usually added by initialize_grid, 
    # but we mock them here for the unit test)
    grid_data.update({"dx": 0.5, "dy": 0.5, "dz": 0.5})
    
    constants = compute_derived_constants(
        grid_data, 
        dummy["fluid_properties"], 
        dummy["simulation_parameters"]
    )
    
    assert "rho" in constants
    assert constants["dt"] == dummy["simulation_parameters"]["time_step"]

def test_allocate_fields_error_handling():
    """Triggers safety gate by passing an invalid dictionary."""
    config = solver_input_schema_dummy()
    
    # Corrupt the grid to trigger ValueError/KeyError branches
    invalid_grid = {"nx": -1, "ny": 2, "nz": 2}
    
    with pytest.raises((ValueError, ZeroDivisionError, KeyError)):
        allocate_fields(invalid_grid)