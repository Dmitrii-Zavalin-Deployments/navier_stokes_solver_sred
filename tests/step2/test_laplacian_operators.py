# tests/step2/test_laplacian_operators.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.create_fluid_mask import create_fluid_mask


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    """
    Construct a minimal valid SolverState for Laplacian operator tests.
    """
    state = SolverState()

    # Grid
    state.grid = type(
        "Grid", (), {"nx": nx, "ny": ny, "nz": nz, "dx": dx, "dy": dx, "dz": dx}
    )()

    # Minimal valid mask (all fluid)
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Compute fluid masks
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Staggered velocity fields
    state.fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    return state


# ------------------------------------------------------------
# 1. Constant field → Laplacian = 0
# ------------------------------------------------------------
def test_laplacian_constant_field():
    state = make_state()

    state.fields["U"].fill(5.0)
    lap_u, lap_v, lap_w = build_laplacian_operators(state)

    lap = lap_u(state.fields["U"])
    assert np.allclose(lap, 0.0)


# ------------------------------------------------------------
# 2. Linear field → Laplacian = 0
# ------------------------------------------------------------
def test_laplacian_linear_field():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)

    # U[i] = i * dx → second derivative = 0
    for i in range(nx + 1):
        state.fields["U"][i, :, :] = i * dx

    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    lap = lap_u(state.fields["U"])

    assert np.allclose(lap, 0.0, atol=1e-6)


# ------------------------------------------------------------
# 3. Quadratic field → Laplacian = constant
# ------------------------------------------------------------
def test_laplacian_quadratic_field():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)

    # U[i] = (i*dx)^2 → second derivative = 2
    for i in range(nx + 1):
        state.fields["U"][i, :, :] = (i * dx) ** 2

    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    lap = lap_u(state.fields["U"])

    # Expected Laplacian = d²/dx² = 2
    assert np.allclose(lap, 2.0, atol=1e-6)


# ------------------------------------------------------------
# 4. Solid neighbors → stencil truncated
# ------------------------------------------------------------
def test_laplacian_solid_neighbors_truncated():
    state = make_state()

    # Mark a neighbor solid
    state.is_fluid[1, 1, 1] = False

    # Give U a simple pattern
    state.fields["U"].fill(10.0)

    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    lap = lap_u(state.fields["U"])

    # Solid cell → Laplacian must be zero
    assert lap[1, 1, 1] == 0.0


# ------------------------------------------------------------
# 5. Boundary-fluid (mask = -1) treated as fluid
# ------------------------------------------------------------
def test_laplacian_boundary_fluid_treated_as_fluid():
    state = make_state()

    # Mark a cell as boundary-fluid
    state.mask[2, 2, 2] = -1
    state.is_fluid, state.is_boundary_cell = create_fluid_mask(state)

    # Give U a simple pattern
    state.fields["U"].fill(3.0)

    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    lap = lap_u(state.fields["U"])

    # Boundary-fluid is treated as fluid → Laplacian finite
    assert np.isfinite(lap[2, 2, 2])


# ------------------------------------------------------------
# 6. Minimal grid (1×1×1) → no index errors
# ------------------------------------------------------------
def test_laplacian_minimal_grid():
    state = make_state(nx=1, ny=1, nz=1)

    lap_u, lap_v, lap_w = build_laplacian_operators(state)
    lap = lap_u(state.fields["U"])

    assert lap.shape == (2, 1, 1)
    assert np.all(np.isfinite(lap))
