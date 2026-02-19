# tests/step1/test_map_geometry_mask_math.py

import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

# ---------------------------------------------------------
# Shape validation (via domain dict)
# ---------------------------------------------------------

def test_invalid_shape_raises():
    """Verifies that malformed domain dictionaries trigger errors."""
    # Negative dimension
    with pytest.raises(ValueError, match="positive"):
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
    with pytest.raises(TypeError, match="iterable"):
        map_geometry_mask(12345, {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Length validation
# ---------------------------------------------------------

def test_mask_flat_length_match():
    """Checks that flat list length equals nx * ny * nz."""
    domain = {"nx": 2, "ny": 2, "nz": 2}

    # Too short
    with pytest.raises(ValueError, match="length"):
        map_geometry_mask([1, 2], domain)

    # Too long
    with pytest.raises(ValueError, match="length"):
        map_geometry_mask(list(range(10)), domain)


# ---------------------------------------------------------
# Mask entry validation
# ---------------------------------------------------------

def test_mask_entries_must_be_finite_integers():
    """Rejects floats, strings, NaNs, and Infs."""
    bad_values = [1.5, "x", float("nan"), float("inf")]

    for bad in bad_values:
        with pytest.raises((ValueError, TypeError)):
            map_geometry_mask([bad], {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Semantic and Object Integration
# ---------------------------------------------------------

def test_semantic_validation_allows_valid_entries():
    """Confirms valid mask entries are correctly reshaped and stored in SolverState."""
    dummy = solver_input_schema_dummy()
    domain = dummy["domain"]
    flat_mask = dummy["mask"]

    # Act
    arr = map_geometry_mask(flat_mask, domain)
    state = SolverState(mask=arr, grid=domain)

    # Verify object-style access
    assert state.mask.shape == (domain["nx"], domain["ny"], domain["nz"])
    
    # Expected: index = i + nx*(j + ny*k)
    expected = np.array(flat_mask).reshape((domain["nx"], domain["ny"], domain["nz"]), order="F")
    assert np.array_equal(state.mask, expected)


def test_canonical_f_order_mapping():
    """Explicitly tests Fortran-order indexing where 'i' varies fastest."""
    # (nx=2, ny=2, nz=1)
    flat = [10, 20, 30, 40]
    domain = {"nx": 2, "ny": 2, "nz": 1}
    
    arr = map_geometry_mask(flat, domain)
    
    # Check specific indices based on i + nx*j
    assert arr[0, 0, 0] == 10  # i=0, j=0
    assert arr[1, 0, 0] == 20  # i=1, j=0
    assert arr[0, 1, 0] == 30  # i=0, j=1
    assert arr[1, 1, 0] == 40  # i=1, j=1