# tests/step2/test_precompute_constants_and_masks.py

import numpy as np
import pytest

from tests.helpers.step2_schema_dummy_state import SchemaDummyState

from src.step2.precompute_constants import precompute_constants
from src.step2.create_fluid_mask import create_fluid_mask


def make_minimal_mask_state(mask: np.ndarray) -> dict:
    """
    Build the minimal Step‑1‑schema‑compliant state required by create_fluid_mask.
    Only the 'fields.Mask' entry is needed.
    """
    return {
        "fields": {
            "Mask": np.asarray(mask)
        }
    }


# ----------------------------------------------------------------------
# precompute_constants tests (now using SchemaDummyState)
# ----------------------------------------------------------------------

def test_precompute_constants_normal():
    state = SchemaDummyState(2, 2, 2, dx=0.1, dy=0.2, dz=0.3, dt=0.01)
    precompute_constants(state)
    constants = state["constants"]

    assert constants["dx"] == pytest.approx(0.1)
    assert constants["inv_dx"] == pytest.approx(10.0)
    assert constants["inv_dx2"] == pytest.approx(100.0)


def test_precompute_constants_very_small_dx():
    state = SchemaDummyState(1, 1, 1, dx=1e-12, dy=1e-12, dz=1e-12, dt=0.01)
    precompute_constants(state)
    constants = state["constants"]

    assert np.isfinite(constants["inv_dx"])
    assert np.isfinite(constants["inv_dx2"])


def test_precompute_constants_dt_zero_rejected():
    state = SchemaDummyState(1, 1, 1, dt=0.0)
    with pytest.raises(ValueError):
        precompute_constants(state)


def test_precompute_constants_existing_constants_passthrough():
    state = SchemaDummyState(1, 1, 1)
    original = {"dx": 0.1}
    state["constants"] = original

    precompute_constants(state)

    assert state["constants"] is original


# ----------------------------------------------------------------------
# create_fluid_mask tests (minimal Step‑1 schema)
# ----------------------------------------------------------------------

def test_create_fluid_mask_mixed():
    mask = np.array(
        [
            [[0, 1]],
            [[-1, 0]],
        ],
        dtype=int,
    )
    state = make_minimal_mask_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert is_fluid.shape == mask.shape
    assert is_boundary.shape == mask.shape

    # fluid cells
    assert is_fluid[0, 0, 1]
    assert is_fluid[1, 0, 0]

    # boundary-fluid
    assert is_boundary[1, 0, 0]

    # solid
    assert not is_fluid[0, 0, 0]


def test_create_fluid_mask_all_fluid():
    mask = np.ones((2, 2, 2), dtype=int)
    state = make_minimal_mask_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert np.all(is_fluid)
    assert not np.any(is_boundary)


def test_create_fluid_mask_all_boundary_fluid():
    mask = -np.ones((2, 2, 2), dtype=int)
    state = make_minimal_mask_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert np.all(is_fluid)
    assert np.all(is_boundary)


def test_create_fluid_mask_shape_preserved():
    mask = np.zeros((3, 4, 5), dtype=int)
    state = make_minimal_mask_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert is_fluid.shape == (3, 4, 5)
    assert is_boundary.shape == (3, 4, 5)


def test_create_fluid_mask_float_rejected():
    mask = np.ones((2, 2, 2), dtype=float)
    state = make_minimal_mask_state(mask)

    with pytest.raises(ValueError):
        create_fluid_mask(state)
