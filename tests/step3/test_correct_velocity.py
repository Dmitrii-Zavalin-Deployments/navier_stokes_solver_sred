# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def _wire_zero_gradients(state):
    """
    Wire pressure gradient operators that return zero everywhere.
    Shapes must match staggered U/V/W.
    """
    P = state["fields"]["P"]

    def gx(P_in):
        return np.zeros_like(state["fields"]["U"])

    def gy(P_in):
        return np.zeros_like(state["fields"]["V"])

    def gz(P_in):
        return np.zeros_like(state["fields"]["W"])

    state["pressure_gradients"] = {
        "x": {"op": gx},
        "y": {"op": gy},
        "z": {"op": gz},
    }


def _wire_unit_gradients(state):
    """
    Wire pressure gradient operators that return ones everywhere.
    """
    def gx(P_in):
        return np.ones_like(state["fields"]["U"])

    def gy(P_in):
        return np.ones_like(state["fields"]["V"])

    def gz(P_in):
        return np.ones_like(state["fields"]["W"])

    state["pressure_gradients"] = {
        "x": {"op": gx},
        "y": {"op": gy},
        "z": {"op": gz},
    }


def test_zero_gradient():
    """
    With zero pressure gradient, velocities must remain unchanged.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_gradients(s2)

    U_star = np.ones_like(s2["fields"]["U"])
    V_star = np.ones_like(s2["fields"]["V"])
    W_star = np.ones_like(s2["fields"]["W"])
    P_new = np.zeros_like(s2["fields"]["P"])

    U_new, V_new, W_new = correct_velocity(s2, U_star, V_star, W_star, P_new)

    assert np.allclose(U_new, U_star)
    assert np.allclose(V_new, V_star)
    assert np.allclose(W_new, W_star)


def test_solid_mask_zero_faces():
    """
    Faces adjacent to solid cells must be zeroed.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_gradients(s2)

    # Mark a solid cell
    mask = np.array(s2["mask_semantics"]["mask"], copy=True)
    mask[1, 1, 1] = 0
    s2["mask_semantics"]["mask"] = mask
    s2["mask_semantics"]["is_solid"] = (mask == 0)

    U_star = np.ones_like(s2["fields"]["U"])
    V_star = np.ones_like(s2["fields"]["V"])
    W_star = np.ones_like(s2["fields"]["W"])
    P_new = np.zeros_like(s2["fields"]["P"])

    U_new, _, _ = correct_velocity(s2, U_star, V_star, W_star, P_new)

    assert np.any(U_new == 0.0)


def test_fluid_adjacent_faces_zero_when_isolated():
    """
    Faces not adjacent to any fluid cell must be zeroed
    when at least one non-fluid cell exists.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_gradients(s2)

    is_fluid = np.array(s2["mask_semantics"]["is_fluid"], copy=True)
    is_fluid[1, 1, 1] = False
    s2["mask_semantics"]["is_fluid"] = is_fluid

    U_star = np.ones_like(s2["fields"]["U"])
    V_star = np.ones_like(s2["fields"]["V"])
    W_star = np.ones_like(s2["fields"]["W"])
    P_new = np.zeros_like(s2["fields"]["P"])

    U_new, _, _ = correct_velocity(s2, U_star, V_star, W_star, P_new)

    assert np.any(U_new == 0.0)


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    def gx(P):
        return np.zeros((2, 1, 1))

    def gy(P):
        return np.zeros((1, 2, 1))

    def gz(P):
        return np.zeros((1, 1, 2))

    state = {
        "constants": {"rho": 1.0, "dt": 0.1},
        "pressure_gradients": {
            "x": {"op": gx},
            "y": {"op": gy},
            "z": {"op": gz},
        },
        "mask_semantics": {
            "is_solid": np.zeros((1, 1, 1), bool),
            "is_fluid": np.ones((1, 1, 1), bool),
        },
    }

    U_star = np.zeros((2, 1, 1))
    V_star = np.zeros((1, 2, 1))
    W_star = np.zeros((1, 1, 2))
    P_new = np.zeros((1, 1, 1))

    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert U_new.shape == U_star.shape
    assert V_new.shape == V_star.shape
    assert W_new.shape == W_star.shape
