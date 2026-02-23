# tests/step1/test_state_and_masking.py

import pytest
import numpy as np
from src.step1.map_geometry_mask import map_geometry_mask
from src.step1.assemble_simulation_state import assemble_simulation_state
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_data():
    """Provides the canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

# --- GEOMETRY MASKING & RESHAPING ---

def test_mask_reshaping_fortran_order(dummy_data):
    """
    Verifies Fortran-order mapping (i-fastest).
    Index = i + nx * (j + ny * k)
    """
    grid = dummy_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz

    # Probe specific indices to verify spatial alignment
    # index 1 should be [1,0,0], index nx should be [0,1,0]
    for target_index in [0, 1, nx, nx * ny - 1, total_cells - 1]:
        flat = [0] * total_cells
        flat[target_index] = 1
        
        mask_3d, _, _ = map_geometry_mask(flat, grid)

        # Expected unravelling logic
        k_exp = target_index // (nx * ny)
        remainder = target_index % (nx * ny)
        j_exp = remainder // nx
        i_exp = remainder % nx

        assert mask_3d[i_exp, j_exp, k_exp] == 1, f"Mapping failed at index {target_index}"

def test_mask_validation_triggers(dummy_data):
    """Triggers length mismatch, unauthorized values, and type pollution."""
    grid = dummy_data["grid"]
    total_cells = grid["nx"] * grid["ny"] * grid["nz"]

    # 1. Length Mismatch
    with pytest.raises(ValueError, match="Mask length mismatch"):
        map_geometry_mask([1, 0, 1], grid)

    # 2. Unauthorized Values (Outside -1, 0, 1)
    bad_val_flat = [0] * total_cells
    bad_val_flat[0] = 5
    with pytest.raises(ValueError, match="Mask contains unauthorized values"):
        map_geometry_mask(bad_val_flat, grid)

    # 3. Type Pollution
    corrupt_flat = [0] * total_cells
    corrupt_flat[0] = "corrupt"
    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask(corrupt_flat, grid)

# --- STATE ASSEMBLY DEBT ---

def test_assemble_state_spatial_incoherence():
    """Trigger: Spatial Incoherence (Line 58)."""
    grid = {"nx": 4, "ny": 4, "nz": 4}
    constants = {"rho": 1.0, "mu": 0.1, "dt": 0.1, "dx": 0.5, "dy": 0.5, "dz": 0.5}
    fields = {
        "U": np.zeros((5, 4, 4)), 
        "V": np.zeros((4, 5, 4)), 
        "W": np.zeros((4, 4, 5)), 
        "P": np.zeros((4, 4, 4))
    }
    mask_small = np.zeros((2, 2, 2)) # Incorrect for 4x4x4 grid
    
    with pytest.raises(ValueError, match="Spatial Incoherence"):
        assemble_simulation_state(
            config={}, 
            grid=grid, 
            fields=fields, 
            mask=mask_small, 
            constants=constants, 
            boundary_conditions={}, 
            is_fluid=mask_small, 
            is_boundary_cell=mask_small
        )

# --- INTEGRATION ---

def test_mask_encapsulation_in_solver_state(dummy_data):
    """Integration: Verifies SolverState correctly holds the 3D mask array."""
    grid = dummy_data["grid"]
    mask_array, _, _ = map_geometry_mask(dummy_data["mask"], grid)
    
    state = SolverState(mask=mask_array, grid=grid)
    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    assert np.issubdtype(state.mask.dtype, np.integer)