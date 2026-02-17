# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _wire_zero_gradients(state):
    """
    Wire pressure gradient operators that return zero everywhere.
    Shapes must match staggered U/V/W.
    """
    def gx(P_in):
        return np.zeros_like(state.fields["U"])

    def gy(P_in):
        return np.zeros_like(state.fields["V"])

    def gz(P_in):
        return np.zeros_like(state.fields["W"])

    state.operators["grad_x"] = gx
    state.operators["grad_y"] = gy
    state.operators["grad_z"] = gz


def _wire_unit_gradients(state):
    """
    Wire pressure gradient operators that return ones everywhere.
    """
    def gx(P_in):
        return np.ones_like(state.fields["U"])

    def gy(P_in):
        return np.ones_like(state.fields["V"])

    def gz(P_in):
        return np.ones_like(state.fields["W"])

    state.operators["grad_x"] = gx
    state.operators["grad_y"] = gy
    state.operators["grad_z"] = gz


def test_zero_gradient():
    """
    With zero pressure gradient, velocities must remain unchanged.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_gradients(state)

    U_star = np.ones_like(state.fields["U"])
    V_star = np.ones_like(state.fields["V"])
    W_star = np.ones_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert np.allclose(U_new, U_star)
    assert np.allclose(V_new, V_star)
    assert np.allclose(W_new, W_star)


def test_solid_mask_zero_faces():
    """
    Faces adjacent to solid cells must be zeroed.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_gradients(state)

    # Mark a solid cell
    state.is_fluid[1, 1, 1] = False

    U_star = np.ones_like(state.fields["U"])
    V_star = np.ones_like(state.fields["V"])
    W_star = np.ones_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"])

    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert np.any(U_new == 0.0)


def test_fluid_adjacent_faces_zero_when_isolated():
    """
    Faces not adjacent to any fluid cell must be zeroed
    when at least one non-fluid cell exists.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_gradients(state)

    # Make one cell non-fluid
    state.is_fluid[1, 1, 1] = False

    U_star = np.ones_like(state.fields["U"])
    V_star = np.ones_like(state.fields["V"])
    W_star = np.ones_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"])

    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert np.any(U_new == 0.0)


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    def gx(P):
        return np.zeros_like(state.fields["U"])

    def gy(P):
        return np.zeros_like(state.fields["V"])

    def gz(P):
        return np.zeros_like(state.fields["W"])

    state.operators["grad_x"] = gx
    state.operators["grad_y"] = gy
    state.operators["grad_z"] = gz

    U_star = np.zeros_like(state.fields["U"])
    V_star = np.zeros_like(state.fields["V"])
    W_star = np.zeros_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_new)

    assert U_new.shape == U_star.shape
    assert V_new.shape == V_star.shape
    assert W_new.shape == W_star.shape
