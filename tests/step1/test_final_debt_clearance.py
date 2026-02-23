import pytest
import numpy as np
from src.step1.allocate_fields import allocate_fields
from src.step1.assemble_simulation_state import assemble_simulation_state

def test_trigger_allocate_fields_negative_dim():
    """Target: allocate_fields.py line 24"""
    # Trigger the 'any(dim <= 0)' check
    with pytest.raises(ValueError, match="Invalid grid dimensions"):
        allocate_fields({"nx": 2, "ny": -1, "nz": 2})

def test_trigger_assemble_state_missing_field():
    """Target: assemble_simulation_state.py line 53"""
    grid = {"nx": 2, "ny": 2, "nz": 2}
    constants = {"rho": 1.0, "mu": 0.1}
    # Fields missing 'W'
    incomplete_fields = {
        "U": np.zeros((3, 2, 2)), 
        "V": np.zeros((2, 3, 2)), 
        "P": np.zeros((2, 2, 2))
    }
    mask = np.zeros((2, 2, 2))
    
    with pytest.raises(KeyError, match="Genesis Error: Required field 'W' missing"):
        assemble_simulation_state(
            config={}, grid=grid, fields=incomplete_fields, 
            mask=mask, constants=constants, boundary_conditions={}, 
            is_fluid=mask, is_boundary_cell=mask
        )
