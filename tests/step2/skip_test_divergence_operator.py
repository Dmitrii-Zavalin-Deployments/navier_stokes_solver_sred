# tests/step2/test_divergence_operator.py

import numpy as np
import pytest

from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import make_step1_dummy_state


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    """
    Create a canonical Step‑1 dummy state and override only the fields
    relevant for divergence operator tests.
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

    return state


# ------------------------------------------------------------
# 1. Zero velocity field → divergence = 0
# ------------------------------------------------------------
def test_divergence_zero_velocity():
    state = make_state()
    divergence = build_divergence_operator(state)

    div = divergence(state.fields)
    assert np.allclose(div, 0.0)


# ------------------------------------------------------------
# 2. Uniform velocity → divergence = 0
# ------------------------------------------------------------
def test_divergence_uniform_velocity():
    state = make_state()

    state.fields["U"].fill(3.0)
    state.fields["V"].fill(-2.0)
    state.fields["W"].fill(1.5)

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert np.allclose(div, 0.0)


# ------------------------------------------------------------
# 3. Linear U field → divergence ≈ 1/dx
# ------------------------------------------------------------
def test_divergence_linear_u_field():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)

    # U[i] = i * dx → dU/dx = 1
    for i in range(nx + 1):
        state.fields["U"][i, :, :] = i * dx

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert np.allclose(div, 1.0, atol=1e-6)


# ------------------------------------------------------------
# 4. Solid region → divergence zeroed
# ------------------------------------------------------------
def test_divergence_solid_region_zeroed():
    state = make_state()

    # Mark center cell as solid
    state.is_fluid[2, 2, 2] = False

    # Give velocities non-zero values
    state.fields["U"].fill(1.0)
    state.fields["V"].fill(1.0)
    state.fields["W"].fill(1.0)

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert div[2, 2, 2] == 0.0


# ------------------------------------------------------------
# 5. Boundary-fluid cells (mask = -1) treated as fluid
# ------------------------------------------------------------
def test_divergence_boundary_fluid_treated_as_fluid():
    state = make_state()

    # Mark a cell as boundary-fluid
    state.mask[1, 1, 1] = -1
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Give velocities a simple pattern
    state.fields["U"].fill(2.0)
    state.fields["V"].fill(2.0)
    state.fields["W"].fill(2.0)

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert np.isfinite(div[1, 1, 1])


# ------------------------------------------------------------
# 6. No-through mask: isolated fluid cell → divergence = 0
# ------------------------------------------------------------
def test_divergence_no_through_mask():
    state = make_state()

    # All solid except center
    state.is_fluid[:] = False
    state.is_fluid[1, 1, 1] = True

    # Velocities arbitrary
    state.fields["U"].fill(5.0)
    state.fields["V"].fill(5.0)
    state.fields["W"].fill(5.0)

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert div[1, 1, 1] == 0.0


# ------------------------------------------------------------
# 7. Minimal grid (1×1×1) → no index errors
# ------------------------------------------------------------
def test_divergence_minimal_grid():
    state = make_state(nx=1, ny=1, nz=1)

    divergence = build_divergence_operator(state)
    div = divergence(state.fields)

    assert div.shape == (1, 1, 1)
    assert np.allclose(div, 0.0)
