# tests/step2/test_build_advection_structure_math.py

import numpy as np
import pytest

from src.step2.build_advection_structure import build_advection_structure


# ---------------------------------------------------------
# Helper: minimal valid MAC-grid state
# ---------------------------------------------------------

def make_state(U, V, W, scheme="central", dx=1.0, dy=1.0, dz=1.0):
    return {
        "constants": {"dx": dx, "dy": dy, "dz": dz},
        "config": {"simulation": {"advection_scheme": scheme}},
        "fields": {
            "U": U,
            "V": V,
            "W": W,
        },
    }


# ---------------------------------------------------------
# Central scheme: simple derivative test
# ---------------------------------------------------------

def test_central_scheme_simple():
    # Minimal valid MAC grid:
    # U: (nx+1, ny, nz) = (2,1,1)
    # V: (nx, ny+1, nz) = (1,2,1)
    # W: (nx, ny, nz+1) = (1,1,2)

    U = np.array([[[0.0]], [[1.0]]])     # linear in x
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="central")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])

    # central derivative of linear function = constant
    assert adv_u.shape == (2, 1, 1)
    assert out["advection_meta"]["interpolation_scheme"] == "central"


# ---------------------------------------------------------
# Upwind scheme: positive velocity branch
# ---------------------------------------------------------

def test_upwind_positive_velocity():
    U = np.array([[[1.0]], [[2.0]]])     # increasing → positive mean
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (2, 1, 1)
    assert out["advection_meta"]["interpolation_scheme"] == "upwind"


# ---------------------------------------------------------
# Upwind scheme: negative velocity branch
# ---------------------------------------------------------

def test_upwind_negative_velocity():
    U = np.array([[[3.0]], [[1.0]]])     # decreasing → negative mean
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (2, 1, 1)


# ---------------------------------------------------------
# Degenerate dimension: nx = 1 triggers early return
# ---------------------------------------------------------

def test_upwind_degenerate_dimension():
    # nx = 0 → U shape = (1,1,1)
    U = np.array([[[1.0]]])
    V = np.array([[[1.0], [1.0]]])       # (1,2,1)
    W = np.array([[[1.0, 1.0]]])         # (1,1,2)

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (1, 1, 1)
    assert adv_u[0, 0, 0] == 0.0


# ---------------------------------------------------------
# Output must be JSON-safe lists
# ---------------------------------------------------------

def test_output_is_json_safe():
    U = np.zeros((2, 1, 1))
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))

    state = make_state(U.tolist(), V.tolist(), W.tolist())
    out = build_advection_structure(state)

    assert isinstance(out["advection"]["u"], list)
    assert isinstance(out["advection"]["v"], list)
    assert isinstance(out["advection"]["w"], list)
