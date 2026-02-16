import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.orchestrate_step2 import orchestrate_step2


def make_state(*, dx=1.0, dy=1.0, dz=1.0, dt=0.1, rho=1.0):
    """
    Construct a minimal valid SolverState for constant precomputation tests.
    Step 2 expects Step 1 to have produced:
      - grid (with dx, dy, dz)
      - config (with dt)
      - constants (with rho)
      - mask (valid tri-state)
      - fields (P, U, V, W)
    """
    state = SolverState()

    # Grid
    state.grid = type("Grid", (), {"dx": dx, "dy": dy, "dz": dz})()

    # Config
    state.config = type("Config", (), {"dt": dt})()

    # Constants (Step 1 normally fills these)
    state.constants = {"rho": rho}

    # Minimal valid mask (1 fluid cell)
    state.mask = np.ones((1, 1, 1), dtype=int)

    # Minimal fields
    state.fields = {
        "P": np.zeros((1, 1, 1)),
        "U": np.zeros((2, 1, 1)),
        "V": np.zeros((1, 2, 1)),
        "W": np.zeros((1, 1, 2)),
    }

    # Boundary conditions (empty but valid)
    state.boundary_conditions = {}

    # Health block
    state.health = {}

    return state


# ------------------------------------------------------------
# 1. Normal constants
# ------------------------------------------------------------
def test_constants_normal():
    state = make_state(dx=0.1, dy=0.2, dz=0.3, dt=0.01)

    orchestrate_step2(state)

    assert state.constants["inv_dx"] == pytest.approx(10.0)
    assert state.constants["inv_dy"] == pytest.approx(5.0)
    assert state.constants["inv_dz"] == pytest.approx(3.3333333333, rel=1e-6)

    assert state.constants["inv_dx2"] == pytest.approx(100.0)
    assert state.constants["inv_dy2"] == pytest.approx(25.0)
    assert state.constants["inv_dz2"] == pytest.approx(11.1111111111, rel=1e-6)


# ------------------------------------------------------------
# 2. Very small dx
# ------------------------------------------------------------
def test_constants_very_small_dx():
    state = make_state(dx=1e-12, dy=1e-12, dz=1e-12)

    orchestrate_step2(state)

    assert np.isfinite(state.constants["inv_dx"])
    assert np.isfinite(state.constants["inv_dx2"])


# ------------------------------------------------------------
# 3. dt = 0 rejected
# ------------------------------------------------------------
def test_constants_dt_zero_rejected():
    state = make_state(dt=0.0)

    with pytest.raises(ValueError):
        orchestrate_step2(state)


# ------------------------------------------------------------
# 4. Constants already present (passthrough)
# ------------------------------------------------------------
def test_constants_existing_passthrough():
    state = make_state()
    state.constants["inv_dx"] = 123.0  # Pretend Step 1 already computed it

    orchestrate_step2(state)

    # Should not overwrite existing values
    assert state.constants["inv_dx"] == 123.0
