# tests/step1/test_geometry_mask.py

import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_data():
    """Provides the canonical dummy input."""
    return solver_input_schema_dummy()

def test_perfect_reshape(dummy_data):
    """Verifies that the dummy's flat mask is reshaped to the dummy's grid dimensions."""
    flat_mask = dummy_data["mask"]
    grid = dummy_data["grid"]
    
    mask = map_geometry_mask(flat_mask, grid)

    assert mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    assert np.issubdtype(mask.dtype, np.integer)

def test_length_mismatch(dummy_data):
    """Verifies that a list with an incorrect number of elements triggers a ValueError."""
    grid = dummy_data["grid"]
    # Create a list that is intentionally too short
    short_flat = [1] * (grid["nx"] * grid["ny"] * grid["nz"] - 1)

    with pytest.raises(ValueError, match="length"):
        map_geometry_mask(short_flat, grid)

def test_data_type_pollution(dummy_data):
    """Ensures Step 1 rejects non-numeric/string data in the mask list."""
    grid = dummy_data["grid"]
    total_cells = grid["nx"] * grid["ny"] * grid["nz"]
    
    bad_flat = [0] * total_cells
    bad_flat[0] = "corrupt_string" 

    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask(bad_flat, grid)

def test_flattening_order_round_trip(dummy_data):
    """
    Verifies that the 3D indexing [i, j, k] matches the canonical 
    flattening rule (Fortran Order): index = i + nx * (j + ny * k).
    
    Uses valid mask values (-1, 0, 1) to satisfy core validation.
    """
    grid = dummy_data["grid"]
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    total_cells = nx * ny * nz

    # We iterate through a few key indices to ensure the mapping is correct
    # without violating the [-1, 0, 1] rule of the core validator.
    for target_index in [0, 1, nx, nx * ny - 1, total_cells - 1]:
        # Create a valid mask of all 0s
        flat = [0] * total_cells
        # Place a '1' at the specific flat index we are testing
        flat[target_index] = 1
        
        mask_3d = map_geometry_mask(flat, grid)

        # Calculate expected 3D coordinates for Fortran order
        # i changes fastest, then j, then k
        k_exp = target_index // (nx * ny)
        remainder = target_index % (nx * ny)
        j_exp = remainder // nx
        i_exp = remainder % nx

        # Verify the '1' ended up in the right 3D spot
        assert mask_3d[i_exp, j_exp, k_exp] == 1, f"Index {target_index} mapped to wrong 3D coordinate."

def test_mask_in_solver_state(dummy_data):
    """
    Integration test: Verifies that the SolverState object holds 
    the mask correctly as an attribute (.mask).
    """
    grid = dummy_data["grid"]
    flat_mask = dummy_data["mask"]
    
    mask_array = map_geometry_mask(flat_mask, grid)
    
    # Instantiate SolverState using Object Style
    # Note: We provide empty dicts for other required fields if necessary
    state = SolverState(
        mask=mask_array, 
        grid=grid, 
        fields={}, 
        constants={}, 
        config={}
    )
    
    # Assertions using attribute access
    assert isinstance(state.mask, np.ndarray)
    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])