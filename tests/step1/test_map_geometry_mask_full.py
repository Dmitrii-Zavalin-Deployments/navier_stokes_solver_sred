# tests/step1/test_map_geometry_mask_full.py

import numpy as np
import pytest
from src.step1.map_geometry_mask import map_geometry_mask
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def dummy_input():
    return solver_input_schema_dummy()

def test_map_geometry_mask_order_integrity(dummy_input):
    """
    Verifies that the flat mask list is reshaped into a 3D array 
    matching the solver's coordinate convention (i, j, k).
    """
    # Use dummy dimensions
    domain = dummy_input["domain"]
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
    
    # Create a unique-value flat list to verify exact placement
    flat = list(range(nx * ny * nz))

    # Act
    mask = map_geometry_mask(flat, domain)
    state = SolverState(mask=mask, grid=domain)

    # Expected: order="F" (Fortran-style) means i varies fastest, then j, then k.
    # This matches the rule: index = i + nx * (j + ny * k)
    expected = np.array(flat).reshape((nx, ny, nz), order="F")

    # 1. Assertions on the array itself
    assert state.mask.shape == (nx, ny, nz)
    assert np.array_equal(state.mask, expected)

    # 2. Manual verification of the first and last elements
    # state.mask[i, j, k]
    assert state.mask[0, 0, 0] == flat[0]
    assert state.mask[nx-1, ny-1, nz-1] == flat[-1]

def test_mask_value_types(dummy_input):
    """Ensures the mask is stored as an integer type in the SolverState."""
    domain = dummy_input["domain"]
    mask = map_geometry_mask(dummy_input["mask"], domain)
    state = SolverState(mask=mask)

    # Mask must be integer to handle categorical values -1, 0, 1
    assert np.issubdtype(state.mask.dtype, np.integer)