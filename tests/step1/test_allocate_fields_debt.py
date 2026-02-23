import pytest
import numpy as np
from src.step1.assemble_simulation_state import assemble_simulation_state

def test_line_58_mismatched_mask():
    """Trigger: Spatial Incoherence (Line 58)."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1}
    # Provide all required fields so we pass the Line 53 check
    fields = {
        "U": np.zeros((5, 4, 4)), 
        "V": np.zeros((4, 5, 4)), 
        "W": np.zeros((4, 4, 5)), 
        "P": np.zeros((4, 4, 4))
    }
    mask = np.zeros((2, 2, 2)) # Incorrect shape for a 4x4x4 grid
    
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state(
            config={}, 
            grid=grid, 
            fields=fields, 
            mask=mask, 
            constants=constants, 
            boundary_conditions={}, 
            is_fluid=mask, 
            is_boundary_cell=mask
        )