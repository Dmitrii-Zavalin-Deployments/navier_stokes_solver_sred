import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.enforce_mask_semantics import enforce_mask_semantics


def make_state(mask: np.ndarray) -> SolverState:
    """
    Construct a minimal valid SolverState for mask semantics tests.
    Only grid + mask are required for enforce_mask_semantics.
    """
    nx, ny, nz = mask.shape

    state = SolverState()
    state.grid = type("Grid", (), {"nx": nx, "ny": ny, "nz": nz})()
    state.mask = mask
    return state


# ------------------------------------------------------------
# 1. Valid tri-state mask
# ------------------------------------------------------------
def test_mask_semantics_valid_tristate():
    mask = np.array([[[1, 0, -1]]], dtype=int)
    state = make_state(mask)

    # Should not raise
    enforce_mask_semantics(state)

    assert state.is_fluid is not None
    assert state.is_boundary_cell is not None
    assert state.is_fluid.shape == mask.shape
    assert state.is_boundary_cell.shape == mask.shape


# ------------------------------------------------------------
# 2. Invalid positive value
# ------------------------------------------------------------
def test_mask_semantics_invalid_positive():
    mask = np.array([[[2]]], dtype=int)
    state = make_state(mask)

    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


# ------------------------------------------------------------
# 3. Invalid negative value
# ------------------------------------------------------------
def test_mask_semantics_invalid_negative():
    mask = np.array([[[-5]]], dtype=int)
    state = make_state(mask)

    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


# ------------------------------------------------------------
# 4. Mixed invalid values
# ------------------------------------------------------------
def test_mask_semantics_mixed_invalid():
    mask = np.array([[[0, 1, -1, 2, 3]]], dtype=int)
    state = make_state(mask)

    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


# ------------------------------------------------------------
# 5. No fluid cells (all zeros)
# ------------------------------------------------------------
def test_mask_semantics_no_fluid_cells():
    mask = np.zeros((2, 2, 2), dtype=int)
    state = make_state(mask)

    with pytest.raises(ValueError):
        enforce_mask_semantics(state)


# ------------------------------------------------------------
# 6. Float mask rejected
# ------------------------------------------------------------
def test_mask_semantics_float_mask_rejected():
    mask = np.array([[[1.0, 0.0, -1.0]]], dtype=float)
    state = make_state(mask)

    with pytest.raises(ValueError):
        enforce_mask_semantics(state)
