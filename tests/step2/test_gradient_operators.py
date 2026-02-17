# tests/step2/test_gradient_operators.py

import numpy as np
import pytest

from src.step2.build_gradient_operators import build_gradient_operators
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    """
    Create a canonical Step‑1 dummy state and override only the fields
    relevant for gradient operator tests.
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

    # Override pressure field (dummy already has correct shape)
    state.fields["P"] = np.zeros((nx, ny, nz))

    return state


# ------------------------------------------------------------
# 1. Constant pressure → gradients = 0
# ------------------------------------------------------------
def test_gradient_constant_pressure():
    state = make_state()
    state.fields["P"].fill(5.0)

    grad_x, grad_y, grad_z = build_gradient_operators(state)

    gx = grad_x(state.fields["P"])
    gy = grad_y(state.fields["P"])
    gz = grad_z(state.fields["P"])

    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)


# ------------------------------------------------------------
# 2. Linear pressure in X → gradient ≈ 1/dx
# ------------------------------------------------------------
def test_gradient_linear_pressure_x():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)

    # P[i] = i * dx → dP/dx = 1
    for i in range(nx):
        state.fields["P"][i, :, :] = i * dx

    grad_x, grad_y, grad_z = build_gradient_operators(state)
    gx = grad_x(state.fields["P"])

    assert np.allclose(gx, 1.0, atol=1e-6)


# ------------------------------------------------------------
# 3. Solid pressure spike → gradient masked or zeroed
# ------------------------------------------------------------
def test_gradient_solid_pressure_spike_zeroed():
    state = make_state()

    # Solid cell in the center
    state.is_fluid[2, 2, 2] = False

    # Pressure spike in that solid cell
    state.fields["P"][2, 2, 2] = 100.0

    grad_x, grad_y, grad_z = build_gradient_operators(state)
    gx = grad_x(state.fields["P"])

    # Gradient must be zero in solid cells
    assert gx[2, 2, 2] == 0.0


# ------------------------------------------------------------
# 4. Output shapes (staggered)
# ------------------------------------------------------------
def test_gradient_output_shapes():
    nx, ny, nz = 3, 4, 5
    state = make_state(nx, ny, nz)

    grad_x, grad_y, grad_z = build_gradient_operators(state)

    gx = grad_x(state.fields["P"])
    gy = grad_y(state.fields["P"])
    gz = grad_z(state.fields["P"])

    assert gx.shape == (nx + 1, ny, nz)
    assert gy.shape == (nx, ny + 1, nz)
    assert gz.shape == (nx, ny, nz + 1)


# ------------------------------------------------------------
# 5. Minimal grid (1×1×1) → no index errors
# ------------------------------------------------------------
def test_gradient_minimal_grid():
    state = make_state(nx=1, ny=1, nz=1)

    grad_x, grad_y, grad_z = build_gradient_operators(state)

    gx = grad_x(state.fields["P"])
    gy = grad_y(state.fields["P"])
    gz = grad_z(state.fields["P"])

    assert gx.shape == (2, 1, 1)
    assert gy.shape == (1, 2, 1)
    assert gz.shape == (1, 1, 2)

    assert np.all(np.isfinite(gx))
    assert np.all(np.isfinite(gy))
    assert np.all(np.isfinite(gz))
