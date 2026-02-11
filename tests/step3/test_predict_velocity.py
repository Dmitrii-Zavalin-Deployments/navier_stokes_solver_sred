# tests/step3/test_predict_velocity.py

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


def _make_fields(s2):
    return {
        "U": np.asarray(s2["fields"]["U"]),
        "V": np.asarray(s2["fields"]["V"]),
        "W": np.asarray(s2["fields"]["W"]),
        "P": np.asarray(s2["fields"]["P"]),
    }


def test_zero_ops_zero_forces():
    """
    With zero advection, zero diffusion, and zero forces,
    predicted velocity must equal the input velocity.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_ops(s2)

    fields = _make_fields(s2)

    U_star, V_star, W_star = predict_velocity(s2, fields)

    assert np.allclose(U_star, fields["U"])
    assert np.allclose(V_star, fields["V"])
    assert np.allclose(W_star, fields["W"])


def test_constant_force():
    """
    Constant external force must modify velocity.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_ops(s2)

    s2["config"]["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    fields = _make_fields(s2)
    U0 = fields["U"].copy()

    U_star, V_star, W_star = predict_velocity(s2, fields)

    assert np.any(U_star > U0)
    assert np.allclose(V_star, fields["V"])
    assert np.allclose(W_star, fields["W"])


def test_solid_mask_respected():
    """
    Faces adjacent to solids must be zeroed.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_ops(s2)

    # Mark a solid cell
    mask = np.array(s2["mask_semantics"]["mask"], copy=True)
    mask[1, 1, 1] = 0
    s2["mask_semantics"]["mask"] = mask
    s2["mask_semantics"]["is_solid"] = (mask == 0)

    s2["config"]["external_forces"] = {"fx": 1.0}

    fields = _make_fields(s2)

    U_star, _, _ = predict_velocity(s2, fields)

    assert np.any(U_star == 0.0)


def test_input_not_mutated():
    """
    predict_velocity must not mutate the input fields.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _wire_zero_ops(s2)

    fields = _make_fields(s2)
    U_before = fields["U"].copy()

    predict_velocity(s2, fields)

    assert np.allclose(fields["U"], U_before)


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    def adv_u(U, V, W):
        return np.zeros((2, 1, 1))

    def adv_v(U, V, W):
        return np.zeros((1, 2, 1))

    def adv_w(U, V, W):
        return np.zeros((1, 1, 2))

    def lap_u(U):
        return np.zeros((2, 1, 1))

    def lap_v(V):
        return np.zeros((1, 2, 1))

    def lap_w(W):
        return np.zeros((1, 1, 2))

    state = {
        "constants": {"rho": 1.0, "mu": 0.1, "dt": 0.01},
        "config": {"external_forces": {}},
        "mask_semantics": {
            "is_solid": np.zeros((1, 1, 1), bool),
        },
        "advection": {
            "u": {"op": adv_u},
            "v": {"op": adv_v},
            "w": {"op": adv_w},
        },
        "laplacians": {
            "u": {"op": lap_u},
            "v": {"op": lap_v},
            "w": {"op": lap_w},
        },
    }

    fields = {
        "U": np.zeros((2, 1, 1)),
        "V": np.zeros((1, 2, 1)),
        "W": np.zeros((1, 1, 2)),
        "P": np.zeros((1, 1, 1)),
    }

    U_star, V_star, W_star = predict_velocity(state, fields)

    assert U_star.shape == fields["U"].shape
    assert V_star.shape == fields["V"].shape
    assert W_star.shape == fields["W"].shape
