# tests/step1/test_assemble_state_debt.py

import pytest
import numpy as np
from src.step1.assemble_simulation_state import assemble_simulation_state

class MockState:
    def __init__(self, fields, mask):
        self.fields = fields
        self.mask = mask
        self.ready_for_time_loop = False

def test_line_53_missing_field():
    """Trigger: Required field 'U' is missing."""
    grid = {"nx": 2, "ny": 2, "nz": 2}
    constants = {"rho": 1.0, "mu": 0.1, "dt": 0.1, "dx": 0.5, "dy": 0.5, "dz": 0.5}
    # Missing "U"
    fields = {"V": np.zeros((2,3,2)), "W": np.zeros((2,2,3)), "P": np.zeros((2,2,2))}
    mask = np.zeros((2, 2, 2))
    state = MockState(fields, mask)
    
    with pytest.raises(KeyError, match="Required field 'U' missing"):
        assemble_simulation_state(
            state, grid, constants, 
            mask=mask, 
            constants=constants, 
            boundary_conditions={}, 
            is_fluid=mask, 
            is_boundary_cell=mask
        )

def test_line_58_mismatched_mask():
    """Trigger: Mask shape mismatch."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1, "dt": 0.1, "dx": 0.5, "dy": 0.5, "dz": 0.5}
    fields = {"U": np.zeros((5,4,4)), "V": np.zeros((4,5,4)), "W": np.zeros((4,4,5)), "P": np.zeros((4,4,4))}
    mask = np.zeros((2, 2, 2)) # Wrong shape
    state = MockState(fields, mask)
    
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state(
            state, grid, constants, 
            mask=mask, 
            constants=constants, 
            boundary_conditions={}, 
            is_fluid=mask, 
            is_boundary_cell=mask
        )