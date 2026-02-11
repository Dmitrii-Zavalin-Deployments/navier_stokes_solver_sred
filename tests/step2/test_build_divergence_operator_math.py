# tests/step2/test_build_divergence_operator_math.py

import numpy as np
import pytest

from src.step2.build_divergence_operator import build_divergence_operator


# ---------------------------------------------------------
# Helper: minimal valid Step‑2 state
# ---------------------------------------------------------

def make_state(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0):
    """
    Produces a minimal valid Step‑2 state with:
      - grid: (nx, ny, nz)
      - mask_3d: fluid everywhere
      - constants: dx, dy, dz
    """
    return {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "constants": {"dx": dx, "dy": dy, "dz": dz},
        "mask_3d": np.ones((nx, ny, nz), dtype=int).tolist(),
    }


# ---------------------------------------------------------
# Test 1 — Basic divergence on minimal MAC grid
# ---------------------------------------------------------

def test_divergence_minimal_grid_zero_fields():
    """
    U, V, W all zero → divergence must be zero.
    """
    state = make_state(nx=1, ny=1, nz=1)
    div_op = build_divergence_operator(state)

    U = np.zeros((2, 1, 1))   # (nx+1, ny, nz)
    V = np.zeros((1, 2, 1))   # (nx, ny+1, nz)
    W = np.zeros((1, 1, 2))   # (nx, ny, nz+1)

    div = div_op(U, V, W)
    assert div.shape == (1, 1, 1)
    assert div[0, 0, 0] == 0.0


# ---------------------------------------------------------
# Test 2 — Simple 1D divergence in x-direction
# ---------------------------------------------------------

def test_divergence_simple_x():
    """
    U increases linearly → constant positive divergence.
    """
    state = make_state(nx=1, ny=1, nz=1, dx=1.0)
    div_op = build_divergence_operator(state)

    U = np.array([[[0.0]], [[2.0]]])   # difference = 2
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    div = div_op(U, V, W)
    assert div[0, 0, 0] == pytest.approx(2.0)


# ---------------------------------------------------------
# Test 3 — Simple 1D divergence in y-direction
# ---------------------------------------------------------

def test_divergence_simple_y():
    state = make_state(nx=1, ny=1, nz=1, dy=2.0)
    div_op = build_divergence_operator(state)

    U = np.zeros((2, 1, 1))
    V = np.array([[[0.0], [4.0]]])     # difference = 4
    W = np.zeros((1, 1, 2))

    div = div_op(U, V, W)
    assert div[0, 0, 0] == pytest.approx(4.0 / 2.0)


# ---------------------------------------------------------
# Test 4 — Simple 1D divergence in z-direction
# ---------------------------------------------------------

def test_divergence_simple_z():
    state = make_state(nx=1, ny=1, nz=1, dz=0.5)
    div_op = build_divergence_operator(state)

    U = np.zeros((2, 1, 1))
    V = np.zeros((1, 2, 1))
    W = np.array([[[1.0, 3.0]]])       # difference = 2

    div = div_op(U, V, W)
    assert div[0, 0, 0] == pytest.approx(2.0 / 0.5)


# ---------------------------------------------------------
# Test 5 — Mask zeroing (solid cell)
# ---------------------------------------------------------

def test_divergence_zeroed_in_solid_cells():
    state = make_state(nx=1, ny=1, nz=1)
    state["mask_3d"] = np.array([[[0]]]).tolist()  # solid cell

    div_op = build_divergence_operator(state)

    U = np.array([[[0.0]], [[10.0]]])
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    div = div_op(U, V, W)
    assert div[0, 0, 0] == 0.0
