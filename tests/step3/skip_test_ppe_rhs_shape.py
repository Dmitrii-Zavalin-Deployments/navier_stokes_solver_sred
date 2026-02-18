# tests/step3/test_ppe_rhs_shape.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _wire_zero_ops(state):
    """
    Wire advection, diffusion, and divergence operators that return zero.
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

    def div(U, V, W):
        return np.zeros_like(state.fields["P"])

    # Step 2 operator API
    state.operators["adv_u"] = adv_u
    state.operators["adv_v"] = adv_v
    state.operators["adv_w"] = adv_w

    state.operators["lap_u"] = lap_u
    state.operators["lap_v"] = lap_v
    state.operators["lap_w"] = lap_w

    state.operators["divergence"] = div


def _make_fields(state):
    return {
        "U": state.fields["U"].copy(),
        "V": state.fields["V"].copy(),
        "W": state.fields["W"].copy(),
        "P": state.fields["P"].copy(),
    }


def test_ppe_rhs_shape():
    """
    RHS of PPE must have the same shape as the pressure field.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Wire zero operators so only shape matters
    _wire_zero_ops(state)

    fields = _make_fields(state)

    # Predict velocity
    U_star, V_star, W_star = predict_velocity(state, fields)

    # Build RHS
    rhs = build_ppe_rhs(state, U_star, V_star, W_star)

    # Shape must match pressure field
    assert rhs.shape == fields["P"].shape
