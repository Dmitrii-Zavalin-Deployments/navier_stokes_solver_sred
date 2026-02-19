# tests/step1/test_map_geometry_mask_math.py

import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask


# ---------------------------------------------------------
# Shape validation (via domain dict)
# ---------------------------------------------------------

def test_invalid_shape_raises():
    # Negative dimension
    with pytest.raises(ValueError):
        map_geometry_mask([1, 2, 3], {"nx": 4, "ny": -1, "nz": 2})

    # Non-integer dimension
    with pytest.raises(ValueError):
        map_geometry_mask([1, 2, 3], {"nx": "bad", "ny": 2, "nz": 2})

    # Missing keys
    with pytest.raises(KeyError):
        map_geometry_mask([1, 2, 3], {"nx": 4, "ny": 4})  # missing nz


# ---------------------------------------------------------
# mask_flat must be iterable
# ---------------------------------------------------------

def test_mask_flat_must_be_iterable():
    with pytest.raises(TypeError):
        map_geometry_mask(12345, {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Length validation
# ---------------------------------------------------------

def test_mask_flat_length_must_match_shape():
    domain = {"nx": 2, "ny": 2, "nz": 2}

    # Too short
    with pytest.raises(ValueError):
        map_geometry_mask([1, 2], domain)

    # Too long
    with pytest.raises(ValueError):
        map_geometry_mask(list(range(10)), domain)


# ---------------------------------------------------------
# Mask entry validation
# ---------------------------------------------------------

def test_mask_entries_must_be_finite_integers():
    bad_values = [1.5, "x", float("nan"), float("inf")]

    for bad in bad_values:
        with pytest.raises(ValueError):
            map_geometry_mask([bad], {"nx": 1, "ny": 1, "nz": 1})


# ---------------------------------------------------------
# Semantic validation (Step 1 allows only {-1, 0, 1})
# ---------------------------------------------------------

def test_semantic_validation_allows_valid_entries():
    flat = [0, 1, -1, 0, 1, 0, -1, 1]
    domain = {"nx": 2, "ny": 2, "nz": 2}

    arr = map_geometry_mask(flat, domain)
    expected = np.array(flat).reshape((2, 2, 2), order="F")

    assert arr.shape == (2, 2, 2)
    assert np.array_equal(arr, expected)


# ---------------------------------------------------------
# Canonical flattening rule tests
# ---------------------------------------------------------

def test_canonical_f_order_mapping():
    flat = [0, 1, -1, 0, 1, 0, -1, 1]
    domain = {"nx": 2, "ny": 2, "nz": 2}

    arr = map_geometry_mask(flat, domain)
    expected = np.array(flat).reshape((2, 2, 2), order="F")

    assert np.array_equal(arr, expected)
