# tests/step1/test_step1_edge_cases.py

import pytest
import numpy as np
from src.step1.parse_config import parse_config
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.allocate_fields import allocate_fields
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_parse_config_defaults_via_inheritance():
    """Triggers fallback logic by stripping optional keys from the dummy."""
    # 1. Inherit the full valid structure
    config = solver_input_schema_dummy()
    
    # 2. Intentionally delete optional keys to trigger fallback defaults in parse_config.py
    if "simulation_parameters" in config:
        del config["simulation_parameters"]
    
    # This should trigger lines like 'if "simulation_parameters" not in data'
    parsed = parse_config(config)
    
    assert "simulation_parameters" in parsed
    # Verify the orchestrator/parser provided its own constitutional defaults
    assert "output_interval" in parsed["simulation_parameters"]

def test_apply_initial_conditions_broadcasting():
    """Triggers array broadcasting logic using dummy values."""
    nx, ny, nz = 2, 2, 2
    fields = {
        "U": np.zeros((nx+1, ny, nz)),
        "V": np.zeros((nx, ny+1, nz)),
        "W": np.zeros((nx, ny, nz+1)),
        "P": np.zeros((nx, ny, nz))
    }
    
    # Use the dummy's initial conditions (scalars/lists) to trigger broadcasting
    dummy_data = solver_input_schema_dummy()
    ic = dummy_data["initial_conditions"] 
    
    # This triggers the broadcast logic: filling (nx+1, ny, nz) with a [0,0,0] vector
    apply_initial_conditions(fields, ic)
    
    assert fields["U"].shape == (3, 2, 2)
    assert np.all(fields["P"] == ic["pressure"])

def test_compute_derived_constants_minimal():
    """Triggers physics derivations by passing dummy sub-dicts."""
    dummy = solver_input_schema_dummy()
    
    state = {"grid": {"dx": 0.5, "dy": 0.5, "dz": 0.5}}
    
    # Passing the specific sub-dicts the function expects
    compute_derived_constants(
        state, 
        dummy["fluid_properties"], 
        dummy["simulation_parameters"]
    )
    
    assert "constants" in state
    assert state["constants"]["rho"] == dummy["fluid_properties"]["density"]

def test_allocate_fields_error_handling():
    """Inherits dummy but corrupts dimensions to trigger Line 24 (ValueErrors)."""
    config = solver_input_schema_dummy()
    # Corruption for edge-case coverage
    config["grid"]["nx"] = -1 
    
    # Testing the safety gate in allocate_fields.py
    with pytest.raises((ValueError, ZeroDivisionError, KeyError)):
        allocate_fields(config["grid"])