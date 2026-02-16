# tests/step1/test_map_geometry_mask_math.py

import numpy as np
import pytest

from src.step1.map_geometry_mask import map_geometry_mask


# ---------------------------------------------------------
# Shape validation
# ---------------------------------------------------------

def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        map_geometry_mask([1, 2, 3], shape=(4, 4), order_formula="C")

    with pytest.raises(ValueError):
        map_geometry_mask([1, 2, 3], shape=(4, -1, 2), order_formula="C")

    with pytest.raises(ValueError):
        map_geometry_mask([1, 2, 3], shape="not-a-shape", order_formula="C")


# ---------------------------------------------------------
# mask_flat must be iterable
# ---------------------------------------------------------

def test_mask_flat_must_be_iterable():
    with pytest.raises(TypeError):
        map_geometry_mask(12345, shape=(1, 1, 1), order_formula="C")


# ---------------------------------------------------------
# Length validation
# ---------------------------------------------------------

def test_mask_flat_length_must_match_shape():
    with pytest.raises(ValueError):
        map_geometry_mask([1, 2], shape=(2, 2, 2), order_formula="C")

    with pytest.raises(ValueError):
        map_geometry_mask(list(range(10)), shape=(2, 2, 2), order_formula="C")


# ---------------------------------------------------------
# Mask entry validation
# ---------------------------------------------------------

def test_mask_entries_must_be_finite_integers():
    bad_values = [1.5, "x", float("nan"), float("inf")]

    for bad in bad_values:
        with pytest.raises(ValueError):   # UPDATED
            map_geometry_mask([bad], shape=(1, 1, 1), order_formula="C")


# ---------------------------------------------------------
# Semantic validation (Step 1 allows arbitrary integers)
# ---------------------------------------------------------

def test_semantic_validation_allows_flattening_test_values():
    arr = map_geometry_mask(
        [0, 1, 2, 3, 4, 5, 6, 7],
        shape=(2, 2, 2),
        order_formula="C"
    )
    assert arr.shape == (2, 2, 2)


# REMOVED:
# test_semantic_validation_rejects_invalid_real_mask_values


# ---------------------------------------------------------
# Flattening order tests
# ---------------------------------------------------------

def test_c_order_mapping():
    flat = list(range(8))
    arr = map_geometry_mask(flat, shape=(2, 2, 2), order_formula="C")
    expected = np.array(flat).reshape((2, 2, 2), order="C")
    assert np.array_equal(arr, expected)


def test_f_order_mapping():
    flat = list(range(8))
    arr = map_geometry_mask(flat, shape=(2, 2, 2), order_formula="F")
    expected = np.array(flat).reshape((2, 2, 2), order="F")
    assert np.array_equal(arr, expected)


def test_fortran_formula_i_nx_j_ny_k():
    flat = list(range(8))
    arr = map_geometry_mask(flat, shape=(2, 2, 2), order_formula="i + nx*(j + ny*k)")
    expected = np.array(flat).reshape((2, 2, 2), order="F")
    assert np.array_equal(arr, expected)


def test_fortran_formula_k_nz_j_ny_i():
    flat = list(range(8))
    arr = map_geometry_mask(flat, shape=(2, 2, 2), order_formula="k + nz*(j + ny*i)")

    tmp = np.array(flat).reshape((2, 2, 2), order="F")
    expected = tmp.transpose(2, 1, 0)

    assert np.array_equal(arr, expected)


def test_fortran_formula_j_ny_i_nx_k():
    flat = list(range(8))
    arr = map_geometry_mask(flat, shape=(2, 2, 2), order_formula="j + ny*(i + nx*k)")

    tmp = np.array(flat).reshape((2, 2, 2), order="F")
    expected = tmp.transpose(1, 0, 2)

    assert np.array_equal(arr, expected)


# ---------------------------------------------------------
# Unknown formula
# ---------------------------------------------------------

def test_unknown_formula_raises():
    with pytest.raises(ValueError):
        map_geometry_mask([0], shape=(1, 1, 1), order_formula="UNKNOWN_ORDER")
