# tests/step2/test_fluid_mask.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.create_fluid_mask import create_fluid_mask


def make_state(mask: np.ndarray) -> SolverState:
    """
    Construct a minimal valid SolverState for fluid-mask tests.
    Only grid + mask are required for create_fluid_mask.
    """
    nx, ny, nz = mask.shape

    state = SolverState()
    state.grid = type("Grid", (), {"nx": nx, "ny": ny, "nz": nz})()
    state.mask = mask
    return state


# ------------------------------------------------------------
# 1. Mixed mask {0, 1, -1}
# ------------------------------------------------------------
def test_fluid_mask_mixed():
    mask = np.array(
        [
            [[0, 1]],
            [[-1, 0]],
        ],
        dtype=int,
    )
    state = make_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert is_fluid.shape == mask.shape
    assert is_boundary.shape == mask.shape

    # Fluid cells: 1 and -1
    assert is_fluid[0, 0, 1]
    assert is_fluid[1, 0, 0]

    # Boundary-fluid: -1 only
    assert is_boundary[1, 0, 0]

    # Solid: 0
    assert not is_fluid[0, 0, 0]


# ------------------------------------------------------------
# 2. All fluid (mask = 1)
# ------------------------------------------------------------
def test_fluid_mask_all_fluid():
    mask = np.ones((2, 2, 2), dtype=int)
    state = make_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert np.all(is_fluid)
    assert not np.any(is_boundary)


# ------------------------------------------------------------
# 3. All boundary-fluid (mask = -1)
# ------------------------------------------------------------
def test_fluid_mask_all_boundary_fluid():
    mask = -np.ones((2, 2, 2), dtype=int)
    state = make_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert np.all(is_fluid)
    assert np.all(is_boundary)


# ------------------------------------------------------------
# 4. Shape preservation
# ------------------------------------------------------------
def test_fluid_mask_shape_preserved():
    mask = np.zeros((3, 4, 5), dtype=int)
    state = make_state(mask)

    is_fluid, is_boundary = create_fluid_mask(state)

    assert is_fluid.shape == (3, 4, 5)
    assert is_boundary.shape == (3, 4, 5)


# ------------------------------------------------------------
# 5. Float mask rejected
# ------------------------------------------------------------
def test_fluid_mask_float_rejected():
    mask = np.ones((2, 2, 2), dtype=float)
    state = make_state(mask)

    with pytest.raises(ValueError):
        create_fluid_mask(state)
