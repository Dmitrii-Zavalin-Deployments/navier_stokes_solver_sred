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
    # Negative dimension: Corrected match to reflect actual core message
    # Added 'r' prefix to fix SyntaxWarning in Python 3.12+
    with pytest.raises(ValueError, match=r"nx\*ny\*nz"):
        map_geometry_mask([1]*8, {"nx": 4, "ny": -1, "nz": 2})

    # Non-integer dimension
    with pytest.raises((ValueError, TypeError)):
        map_geometry_mask([1]*16, {"nx": "bad", "ny": 2, "nz": 2})

    # Missing keys
    with pytest.raises(KeyError):
        map_geometry_mask([1, 2, 3], {"nx": 4, "ny": 4})  # missing nz


# ---------------------------------------------------------
# mask_flat must be iterable
# ---------------------------------------------------------

def test_mask_flat_must_be_iterable():
    """Ensures input mask must be a list or array-like."""
    # Corrected match to align with Python's native error for non-iterables
    with pytest.raises(TypeError, match="type 'int' has no len"):
        map_geometry_mask(12345, {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Length validation
# ---------------------------------------------------------

def test_mask_flat_length_match():
    """Checks that flat list length equals nx * ny * nz."""
    grid = {"nx": 2, "ny": 2, "nz": 2}

    # Too short: Added 'r' prefix to fix SyntaxWarning
    with pytest.raises(ValueError, match=r"match nx\*ny\*nz"):
        map_geometry_mask([1, 2], grid)

    # Too long: Added 'r' prefix to fix SyntaxWarning
    with pytest.raises(ValueError, match=r"match nx\*ny\*nz"):
        map_geometry_mask(list(range(10)), grid)


# ---------------------------------------------------------
# Mask entry validation
# ---------------------------------------------------------

def test_mask_entries_must_be_finite_integers():
    """Rejects floats, strings, NaNs, and Infs."""
    bad_values = [1.5, "x", float("nan"), float("inf")]

    for bad in bad_values:
        # Some bad values trigger TypeError during conversion, others ValueError
        with pytest.raises((ValueError, TypeError)):
            map_geometry_mask([bad], {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Semantic and Object Integration
# ---------------------------------------------------------

def test_semantic_validation_allows_valid_entries():
    """Confirms valid mask entries are correctly reshaped and stored in SolverState."""
    dummy = solver_input_schema_dummy()
    grid = dummy["grid"]
    flat_mask = dummy["mask"]

    # Act
    arr = map_geometry_mask(flat_mask, grid)
    state = SolverState(mask=arr, grid=grid)

    # Verify object-style access
    assert state.mask.shape == (grid["nx"], grid["ny"], grid["nz"])
    
    # Expected: index = i + nx*(j + ny*k) (Fortran Order)
    expected = np.array(flat_mask).reshape((grid["nx"], grid["ny"], grid["nz"]), order="F")
    assert np.array_equal(state.mask, expected)


def test_canonical_f_order_mapping():
    """
    Explicitly tests Fortran-order indexing where 'i' varies fastest.
    Uses only values -1, 0, 1 to pass semantic validation.
    """
    # (nx=2, ny=2, nz=1)
    # Using a specific pattern to ensure indices are mapped correctly:
    # index 0 (0,0): 1
    # index 1 (1,0): 0
    # index 2 (0,1): -1
    # index 3 (1,1): 1
    flat = [1, 0, -1, 1]
    grid = {"nx": 2, "ny": 2, "nz": 1}
    
    arr = map_geometry_mask(flat, grid)
    
    # Check specific indices based on i + nx*j
    assert arr[0, 0, 0] == 1   # i=0, j=0 (First element)
    assert arr[1, 0, 0] == 0   # i=1, j=0 (Second element)
    assert arr[0, 1, 0] == -1  # i=0, j=1 (Third element)
    assert arr[1, 1, 0] == 1   # i=1, j=1 (Fourth element)