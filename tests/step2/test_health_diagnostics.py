# tests/step2/test_health_diagnostics.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.compute_initial_health import compute_initial_health
from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.create_fluid_mask import create_fluid_mask


def make_state(nx=4, ny=4, nz=4, dx=1.0, dt=0.1):
    """
    Construct a minimal valid SolverState for health diagnostics tests.
    """
    state = SolverState()

    # Grid
    state.grid = type(
        "Grid", (), {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dx, "dz": dx}
    )()

    # Config
    state.config = type("Config", (), {"dt": dt})()

    # Mask (all fluid)
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Staggered velocity fields
    state.fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Pressure field (needed for divergence operator)
    state.fields["P"] = np.zeros((nx, ny, nz))

    # Health block
    state.health = {}

    return state


# ------------------------------------------------------------
# 1. Zero velocity → divergence_norm=0, max_vel=0, CFL=0
# ------------------------------------------------------------
def test_health_zero_velocity():
    state = make_state()

    divergence = build_divergence_operator(state)(state.fields)
    compute_initial_health(state, divergence)

    assert state.health["divergence_norm"] == 0.0
    assert state.health["max_velocity"] == 0.0
    assert state.health["cfl"] == 0.0


# ------------------------------------------------------------
# 2. Uniform velocity → CFL = dt*(|u|/dx + |v|/dy + |w|/dz)
# ------------------------------------------------------------
def test_health_uniform_velocity_cfl():
    dx = 0.5
    dt = 0.2
    state = make_state(dx=dx, dt=dt)

    # Uniform velocities
    u0, v0, w0 = 2.0, -1.0, 0.5
    state.fields["U"].fill(u0)
    state.fields["V"].fill(v0)
    state.fields["W"].fill(w0)

    divergence = build_divergence_operator(state)(state.fields)
    compute_initial_health(state, divergence)

    expected_cfl = dt * (abs(u0) / dx + abs(v0) / dx + abs(w0) / dx)
    assert state.health["cfl"] == pytest.approx(expected_cfl)


# ------------------------------------------------------------
# 3. Divergent field → divergence_norm matches L2 norm
# ------------------------------------------------------------
def test_health_divergent_field_l2_norm():
    state = make_state()

    # Divergence field with known L2 norm
    divergence = np.ones((state.grid.nx, state.grid.ny, state.grid.nz))
    expected_norm = np.sqrt(np.sum(divergence**2))

    compute_initial_health(state, divergence)

    assert state.health["divergence_norm"] == pytest.approx(expected_norm)


# ------------------------------------------------------------
# 4. CFL > 1 → still computed, no crash
# ------------------------------------------------------------
def test_health_cfl_greater_than_one():
    dx = 0.1
    dt = 1.0
    state = make_state(dx=dx, dt=dt)

    # Large velocities → CFL > 1
    state.fields["U"].fill(10.0)
    state.fields["V"].fill(10.0)
    state.fields["W"].fill(10.0)

    divergence = build_divergence_operator(state)(state.fields)
    compute_initial_health(state, divergence)

    assert state.health["cfl"] > 1.0
    assert np.isfinite(state.health["cfl"])


# ------------------------------------------------------------
# 5. Extremely small dx → CFL finite, no overflow
# ------------------------------------------------------------
def test_health_extremely_small_dx():
    dx = 1e-12
    dt = 0.1
    state = make_state(dx=dx, dt=dt)

    # Moderate velocities
    state.fields["U"].fill(1.0)
    state.fields["V"].fill(1.0)
    state.fields["W"].fill(1.0)

    divergence = build_divergence_operator(state)(state.fields)
    compute_initial_health(state, divergence)

    assert np.isfinite(state.health["cfl"])
    assert not np.isnan(state.health["cfl"])
