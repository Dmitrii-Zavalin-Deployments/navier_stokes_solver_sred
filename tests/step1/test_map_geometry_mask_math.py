# tests/step1/test_map_geometry_mask_math.py

import numpy as np
import pytest
from src.step1.map_geometry_mask import map_geometry_mask
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# ---------------------------------------------------------
# Shape validation (via grid dict)
# ---------------------------------------------------------

def test_invalid_shape_raises():
    """Verifies that malformed grid dictionaries trigger errors."""
    # Negative dimension
    with pytest.raises(ValueError, match=r"nx\*ny\*nz"):
        map_geometry_mask([1]*8, {"nx": 4, "ny": -1, "nz": 2})

    # Non-integer dimension
    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask([1]*16, {"nx": "bad", "ny": 2, "nz": 2})

    # Missing keys
    with pytest.raises(KeyError):
        map_geometry_mask([1, 2, 3], {"nx": 4, "ny": 4})  


# ---------------------------------------------------------
# mask_flat must be iterable
# ---------------------------------------------------------

def test_mask_flat_must_be_iterable():
    """Ensures input mask must be a list or array-like for length checking."""
    with pytest.raises(TypeError, match=r"has no len|not iterable"):
        map_geometry_mask(12345, {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Length validation
# ---------------------------------------------------------

def test_mask_flat_length_match():
    """Checks that flat list length equals nx * ny * nz."""
    grid = {"nx": 2, "ny": 2, "nz": 2}

    # Length mismatch (too short or too long)
    for bad_list in [[1, 2], list(range(10))]:
        with pytest.raises(ValueError, match=r"match nx\*ny\*nz"):
            map_geometry_mask(bad_list, grid)


# ---------------------------------------------------------
# Mask entry validation
# ---------------------------------------------------------

def test_mask_entries_must_be_finite_integers():
    """Rejects floats, strings, NaNs, and Infs to maintain discrete logic."""
    bad_values = [1.5, "x", float("nan"), float("inf")]

    for bad in bad_values:
        with pytest.raises((ValueError, TypeError)):
            map_geometry_mask([bad], {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Semantic and Object Integration
# ---------------------------------------------------------

def test_semantic_validation_allows_valid_entries():
    """Confirms valid mask entries are correctly reshaped and stored."""
    dummy = solver_input_schema_dummy()
    grid = dummy["grid"]
    flat_mask = dummy["mask"]

    arr, _, _ = map_geometry_mask(flat_mask, grid)
    state = SolverState(mask=arr, grid=grid)

    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    
    # Expected: index = i + nx*(j + ny*k) (Fortran Order)
    expected = np.array(flat_mask).reshape((grid["nx"], grid["ny"], grid["nz"]), order="F")
    assert np.array_equal(state.mask, expected)


def test_canonical_f_order_mapping():
    """
    Explicitly tests Fortran-order indexing where 'i' varies fastest.
    L = i + (nx * j) + (nx * ny * k)
    """
    flat = [1, 0, -1, 1]
    grid = {"nx": 2, "ny": 2, "nz": 1}
    
    arr, _, _ = map_geometry_mask(flat, grid)
    
    assert arr[0, 0, 0] == 1   # i=0, j=0
    assert arr[1, 0, 0] == 0   # i=1, j=0
    assert arr[0, 1, 0] == -1  # i=0, j=1
    assert arr[1, 1, 0] == 1   # i=1, j=1