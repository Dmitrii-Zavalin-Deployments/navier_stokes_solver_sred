# tests/step2/test_build_advection_structure_math.py

import numpy as np
import pytest

from src.step2.build_advection_structure import build_advection_structure


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
    # U increases linearly → derivative = constant
    U = np.array([[0.0], [1.0], [2.0]])
    V = np.zeros((3, 1))
    W = np.zeros((3, 1))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="central")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])

    # central derivative of linear function = constant
    assert adv_u[1, 0] != 0.0  # interior derivative exists
    assert out["advection_meta"]["interpolation_scheme"] == "central"


# ---------------------------------------------------------
# Upwind scheme: positive velocity branch
# ---------------------------------------------------------

def test_upwind_positive_velocity():
    # Positive velocities → use backward difference
    U = np.array([[1.0], [2.0], [3.0]])
    V = np.zeros((3, 1))
    W = np.zeros((3, 1))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (3, 1)
    assert out["advection_meta"]["interpolation_scheme"] == "upwind"


# ---------------------------------------------------------
# Upwind scheme: negative velocity branch
# ---------------------------------------------------------

def test_upwind_negative_velocity():
    # Negative velocities → use forward difference
    U = np.array([[3.0], [2.0], [1.0]])
    V = np.zeros((3, 1))
    W = np.zeros((3, 1))

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (3, 1)


# ---------------------------------------------------------
# Degenerate dimension: nx = 1 triggers early return
# ---------------------------------------------------------

def test_upwind_degenerate_dimension():
    U = np.array([[1.0]])  # nx = 1
    V = np.array([[1.0]])
    W = np.array([[1.0]])

    state = make_state(U.tolist(), V.tolist(), W.tolist(), scheme="upwind")
    out = build_advection_structure(state)

    adv_u = np.array(out["advection"]["u"])
    assert adv_u.shape == (1, 1)
    assert adv_u[0, 0] == 0.0  # early-return zero field


# ---------------------------------------------------------
# Output must be JSON-safe lists
# ---------------------------------------------------------

def test_output_is_json_safe():
    U = np.zeros((2, 1))
    V = np.zeros((2, 1))
    W = np.zeros((2, 1))

    state = make_state(U.tolist(), V.tolist(), W.tolist())
    out = build_advection_structure(state)

    assert isinstance(out["advection"]["u"], list)
    assert isinstance(out["advection"]["v"], list)
    assert isinstance(out["advection"]["w"], list)
