# tests/step3/test_predict_velocity.py

import numpy as np
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _wire_zero_ops(state):
    """
    Wire advection and diffusion operators that return zero everywhere.
    """
    def adv_u(U, V, W):
        return np.zeros_like(U)

    def adv_v(U, V, W):
        return np.zeros_like(V)

    def adv_w(U, V, W):
        return np.zeros_like(W)

    def lap_u(U):
        return np.zeros_like(U)

    def lap_v(V):
        return np.zeros_like(V)

    def lap_w(W):
        return np.zeros_like(W)

    state.operators["adv_u"] = adv_u
    state.operators["adv_v"] = adv_v
    state.operators["adv_w"] = adv_w

    state.operators["lap_u"] = lap_u
    state.operators["lap_v"] = lap_v
    state.operators["lap_w"] = lap_w


def _make_fields(state):
    return {
        "U": state.fields["U"].copy(),
        "V": state.fields["V"].copy(),
        "W": state.fields["W"].copy(),
        "P": state.fields["P"].copy(),
    }


def test_zero_ops_zero_forces():
    """
    With zero advection, zero diffusion, and zero forces,
    predicted velocity must equal the input velocity.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_ops(state)

    fields = _make_fields(state)

    U_star, V_star, W_star = predict_velocity(state, fields)

    assert np.allclose(U_star, fields["U"])
    assert np.allclose(V_star, fields["V"])
    assert np.allclose(W_star, fields["W"])


def test_constant_force():
    """
    Constant external force must modify velocity.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_ops(state)

    state.config["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    fields = _make_fields(state)
    U0 = fields["U"].copy()

    U_star, V_star, W_star = predict_velocity(state, fields)

    assert np.any(U_star > U0)
    assert np.allclose(V_star, fields["V"])
    assert np.allclose(W_star, fields["W"])


def test_solid_mask_respected():
    """
    Faces adjacent to solids must be zeroed.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_ops(state)

    # Mark a solid cell
    state.is_fluid[1, 1, 1] = False

    state.config["external_forces"] = {"fx": 1.0}

    fields = _make_fields(state)

    U_star, _, _ = predict_velocity(state, fields)

    assert np.any(U_star == 0.0)


def test_input_not_mutated():
    """
    predict_velocity must not mutate the input fields.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_zero_ops(state)

    fields = _make_fields(state)
    U_before = fields["U"].copy()

    predict_velocity(state, fields)

    assert np.allclose(fields["U"], U_before)


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    def adv_u(U, V, W):
        return np.zeros_like(U)

    def adv_v(U, V, W):
        return np.zeros_like(V)

    def adv_w(U, V, W):
        return np.zeros_like(W)

    def lap_u(U):
        return np.zeros_like(U)

    def lap_v(V):
        return np.zeros_like(V)

    def lap_w(W):
        return np.zeros_like(W)

    state.operators["adv_u"] = adv_u
    state.operators["adv_v"] = adv_v
    state.operators["adv_w"] = adv_w
    state.operators["lap_u"] = lap_u
    state.operators["lap_v"] = lap_v
    state.operators["lap_w"] = lap_w

    fields = _make_fields(state)

    U_star, V_star, W_star = predict_velocity(state, fields)

    assert U_star.shape == fields["U"].shape
    assert V_star.shape == fields["V"].shape
    assert W_star.shape == fields["W"].shape
