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
    """Verifies that the dummy's flat mask is reshaped to the dummy's domain dimensions."""
    flat_mask = dummy_data["mask"]
    domain = dummy_data["domain"]
    
    mask = map_geometry_mask(flat_mask, domain)

    assert mask.shape == (domain["nx"], domain["ny"], domain["nz"])
    assert np.issubdtype(mask.dtype, np.integer)

def test_length_mismatch(dummy_data):
    """Verifies that a list with an incorrect number of elements triggers a ValueError."""
    domain = dummy_data["domain"]
    # Create a list that is intentionally too short
    short_flat = [1] * (domain["nx"] * domain["ny"] * domain["nz"] - 1)

    with pytest.raises(ValueError, match="length"):
        map_geometry_mask(short_flat, domain)

def test_data_type_pollution(dummy_data):
    """Ensures Step 1 rejects non-numeric/string data in the mask list."""
    domain = dummy_data["domain"]
    total_cells = domain["nx"] * domain["ny"] * domain["nz"]
    
    bad_flat = [0] * total_cells
    bad_flat[0] = "corrupt_string" 

    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask(bad_flat, domain)

def test_flattening_order_round_trip(dummy_data):
    """
    Verifies that the 3D indexing [i, j, k] matches the canonical 
    flattening rule: index = i + nx * (j + ny * k).
    """
    domain = dummy_data["domain"]
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]

    # Generate a flat mask where each value is its own index
    flat = list(range(nx * ny * nz))
    mask = map_geometry_mask(flat, domain)

    # Verify every coordinate maps back correctly to the flat index
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                expected_index = i + nx * (j + ny * k)
                assert mask[i, j, k] == expected_index

def test_mask_in_solver_state(dummy_data):
    """
    Integration test: Verifies that the SolverState object holds 
    the mask correctly as an attribute (.mask).
    """
    domain = dummy_data["domain"]
    flat_mask = dummy_data["mask"]
    
    mask_array = map_geometry_mask(flat_mask, domain)
    
    # Instantiate SolverState (Object Style)
    state = SolverState(mask=mask_array, grid=domain)
    
    # Assertions using attribute access
    assert isinstance(state.mask, np.ndarray)
    assert state.mask.shape == (domain["nx"], domain["ny"], domain["nz"])