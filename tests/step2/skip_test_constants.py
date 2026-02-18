# tests/step2/test_constants.py

import numpy as np
import pytest

from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state


def make_state(*, dx=1.0, dy=1.0, dz=1.0, dt=0.1, rho=1.0):
    """
    Create a canonical Step‑1 dummy state and override only the fields
    relevant for constant precomputation tests.

    Step 2 expects Step 1 to have produced:
      - grid (dx, dy, dz)
      - config (dt)
      - constants (rho)
      - mask (valid tri-state)
      - fields (P, U, V, W)
    """
    # Create a minimal 1×1×1 Step‑1 dummy
    state = make_step1_dummy_state(nx=1, ny=1, nz=1, dx=dx, dy=dy, dz=dz, dt=dt, rho=rho)

    # Override grid spacings (dummy uses dx for all unless overridden)
    state.grid.dx = dx
    state.grid.dy = dy
    state.grid.dz = dz

    # Override dt and rho
    state.config.dt = dt
    state.constants["rho"] = rho

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
