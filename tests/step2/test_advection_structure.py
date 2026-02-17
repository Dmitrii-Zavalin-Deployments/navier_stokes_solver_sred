# tests/step2/test_advection_structure.py

import numpy as np
import pytest

from src.step2.build_advection_structure import build_advection_structure
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state


def make_state(nx=4, ny=4, nz=4, dx=1.0, scheme="upwind"):
    """
    Create a canonical Step‑1 dummy state and override only the fields
    relevant for advection structure tests.
    """
    # Create Step‑1 dummy
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz, dx=dx)

    # Override grid spacing (dummy uses dx for all unless overridden)
    state.grid.dx = dx
    state.grid.dy = dx
    state.grid.dz = dx

    # Ensure mask is all fluid
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Recompute fluid masks
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Override staggered velocity fields (dummy already has correct shapes)
    state.fields["U"] = np.zeros((nx + 1, ny, nz))
    state.fields["V"] = np.zeros((nx, ny + 1, nz))
    state.fields["W"] = np.zeros((nx, ny, nz + 1))

    # Override advection scheme (Step‑1 normally sets only dt)
    state.config.advection_scheme = scheme

    return state


# ------------------------------------------------------------
# 1. Zero velocity → advection = 0
# ------------------------------------------------------------
def test_advection_zero_velocity():
    state = make_state()

    adv_u, adv_v, adv_w = build_advection_structure(state)

    u_adv = adv_u(state.fields)
    v_adv = adv_v(state.fields)
    w_adv = adv_w(state.fields)

    assert np.allclose(u_adv, 0.0)
    assert np.allclose(v_adv, 0.0)
    assert np.allclose(w_adv, 0.0)


# ------------------------------------------------------------
# 2. Uniform velocity → advection = 0
# ------------------------------------------------------------
def test_advection_uniform_velocity():
    state = make_state()

    state.fields["U"].fill(2.0)
    state.fields["V"].fill(-1.0)
    state.fields["W"].fill(0.5)

    adv_u, adv_v, adv_w = build_advection_structure(state)

    u_adv = adv_u(state.fields)
    v_adv = adv_v(state.fields)
    w_adv = adv_w(state.fields)

    assert np.allclose(u_adv, 0.0)
    assert np.allclose(v_adv, 0.0)
    assert np.allclose(w_adv, 0.0)


# ------------------------------------------------------------
# 3. Linear velocity field → non-zero advection
# ------------------------------------------------------------
def test_advection_linear_field():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)

    # U[i] = i → produces non-zero advection
    for i in range(nx + 1):
        state.fields["U"][i, :, :] = float(i)

    adv_u, adv_v, adv_w = build_advection_structure(state)
    u_adv = adv_u(state.fields)

    assert np.any(np.abs(u_adv) > 1e-12)


# ------------------------------------------------------------
# 4. Upwind vs Central → different numerical results
# ------------------------------------------------------------
def test_advection_upwind_vs_central():
    nx, ny, nz = 4, 4, 4

    # Upwind
    state_upwind = make_state(nx, ny, nz, scheme="upwind")
    for i in range(nx + 1):
        state_upwind.fields["U"][i, :, :] = float(i)
    adv_u_upwind, _, _ = build_advection_structure(state_upwind)
    result_upwind = adv_u_upwind(state_upwind.fields)

    # Central
    state_central = make_state(nx, ny, nz, scheme="central")
    for i in range(nx + 1):
        state_central.fields["U"][i, :, :] = float(i)
    adv_u_central, _, _ = build_advection_structure(state_central)
    result_central = adv_u_central(state_central.fields)

    assert not np.allclose(result_upwind, result_central)


# ------------------------------------------------------------
# 5. Upwind on discontinuous field → no oscillations
# ------------------------------------------------------------
def test_advection_upwind_discontinuous_no_oscillations():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz, scheme="upwind")

    # Sharp jump
    state.fields["U"][:2, :, :] = 0.0
    state.fields["U"][2:, :, :] = 10.0

    adv_u, _, _ = build_advection_structure(state)
    u_adv = adv_u(state.fields)

    # Upwind should not produce oscillations (no overshoot)
    assert np.all(u_adv <= 10.0 + 1e-6)
    assert np.all(u_adv >= -1e-6)


# ------------------------------------------------------------
# 6. Minimal grid (1×1×1) → no index errors
# ------------------------------------------------------------
def test_advection_minimal_grid():
    state = make_state(nx=1, ny=1, nz=1)

    adv_u, adv_v, adv_w = build_advection_structure(state)

    u_adv = adv_u(state.fields)
    v_adv = adv_v(state.fields)
    w_adv = adv_w(state.fields)

    assert u_adv.shape == (2, 1, 1)
    assert v_adv.shape == (1, 2, 1)
    assert w_adv.shape == (1, 1, 2)

    assert np.all(np.isfinite(u_adv))
    assert np.all(np.isfinite(v_adv))
    assert np.all(np.isfinite(w_adv))
