# tests/step3/test_ppe_rhs_shape.py

import numpy as np
from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.predict_velocity import predict_velocity
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


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
        return np.zeros_like(state["fields"]["P"])

    state["advection"] = {
        "u": {"op": adv_u},
        "v": {"op": adv_v},
        "w": {"op": adv_w},
    }

    state["laplacians"] = {
        "u": {"op": lap_u},
        "v": {"op": lap_v},
        "w": {"op": lap_w},
    }

    state["divergence"] = {"op": div}


def _make_fields(s2):
    return {
        "U": np.asarray(s2["fields"]["U"]),
        "V": np.asarray(s2["fields"]["V"]),
        "W": np.asarray(s2["fields"]["W"]),
        "P": np.asarray(s2["fields"]["P"]),
    }


def test_ppe_rhs_shape():
    """
    RHS of PPE must have the same shape as the pressure field.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    # Wire zero operators so only shape matters
    _wire_zero_ops(s2)

    fields = _make_fields(s2)

    # Predict velocity
    U_star, V_star, W_star = predict_velocity(s2, fields)

    # Build RHS
    rhs = build_ppe_rhs(s2, U_star, V_star, W_star)

    # Shape must match pressure field
    assert rhs.shape == fields["P"].shape
