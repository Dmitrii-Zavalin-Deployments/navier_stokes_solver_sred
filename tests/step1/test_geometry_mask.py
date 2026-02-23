# tests/step1/test_geometry_mask.py

import numpy as np
import pytest
from src.step1.map_geometry_mask import map_geometry_mask
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_data():
    """Provides the canonical dummy input (Section 5 Compliance)."""
    return solver_input_schema_dummy()

def test_perfect_reshape(dummy_data):
    """Verifies that the dummy's flat mask is reshaped to the dummy's grid dimensions."""
    flat_mask = dummy_data["mask"]
    grid = dummy_data["grid"]
    
    mask, _, _ = map_geometry_mask(flat_mask, grid)

    assert mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    assert np.issubdtype(mask.dtype, np.integer)

def test_length_mismatch(dummy_data):
    """Verifies that a list with an incorrect number of elements triggers a ValueError."""
    grid = dummy_data["grid"]
    short_flat = [1] * (grid["nx"] * grid["ny"] * grid["nz"] - 1)

    with pytest.raises(ValueError, match="(?i)length"):
        map_geometry_mask(short_flat, grid)

def test_data_type_pollution(dummy_data):
    """Ensures Step 1 rejects non-numeric data in the mask list (Debt Prevention)."""
    grid = dummy_data["grid"]
    total_cells = grid["nx"] * grid["ny"] * grid["nz"]
    
    bad_flat = [0] * total_cells
    bad_flat[0] = "corrupt_string" 

    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask(bad_flat, grid)

def test_flattening_order_round_trip(dummy_data):
    """
    Compliance: Verifies Fortran Order (i-fastest).
    Index = i + nx * (j + ny * k)
    """
    grid = dummy_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz

    for target_index in [0, 1, nx, nx * ny - 1, total_cells - 1]:
        flat = [0] * total_cells
        flat[target_index] = 1
        
        mask_3d, _, _ = map_geometry_mask(flat, grid)

        # Fortran unraveling
        k_exp = target_index // (nx * ny)
        remainder = target_index % (nx * ny)
        j_exp = remainder // nx
        i_exp = remainder % nx

        assert mask_3d[i_exp, j_exp, k_exp] == 1, f"Fortran mapping failed at index {target_index}"

def test_mask_in_solver_state(dummy_data):
    """Integration: Verifies SolverState correctly encapsulates the 3D mask."""
    grid = dummy_data["grid"]
    flat_mask = dummy_data["mask"]
    
    mask_array, _, _ = map_geometry_mask(flat_mask, grid)
    
    state = SolverState(
        mask=mask_array, 
        grid=grid, 
        fields={}, 
        constants={}, 
        config={}
    )
    
    assert isinstance(state.mask, np.ndarray)
    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])