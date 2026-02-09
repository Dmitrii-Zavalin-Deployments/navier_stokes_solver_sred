# tests/step3/test_external_forces.py

import numpy as np
from src.step3.predict_velocity import predict_velocity
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


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


def test_external_forces_modify_velocity():
    """
    External forces must modify predicted velocity.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    # Wire zero advection/diffusion so only forces act
    _wire_zero_ops(s2)

    # Apply a simple force in x-direction
    s2["config"]["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    fields = {
        "U": np.asarray(s2["fields"]["U"]),
        "V": np.asarray(s2["fields"]["V"]),
        "W": np.asarray(s2["fields"]["W"]),
        "P": np.asarray(s2["fields"]["P"]),
    }

    U0 = fields["U"].copy()

    U_star, V_star, W_star = predict_velocity(s2, fields)

    # U* must increase due to fx
    assert np.any(U_star > U0)
