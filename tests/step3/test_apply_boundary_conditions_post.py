# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_state_update():
    """
    apply_boundary_conditions_post must return a new fields dict
    containing exactly the fields passed in.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    U_old = np.asarray(s2["fields"]["U"])
    V_old = np.asarray(s2["fields"]["V"])
    W_old = np.asarray(s2["fields"]["W"])
    P_old = np.asarray(s2["fields"]["P"])

    U_new = np.ones_like(U_old)
    V_new = np.ones_like(V_old)
    W_new = np.ones_like(W_old)
    P_new = np.ones_like(P_old)

    fields_out = apply_boundary_conditions_post(s2, U_new, V_new, W_new, P_new)

    assert np.allclose(fields_out["U"], U_new)
    assert np.allclose(fields_out["V"], V_new)
    assert np.allclose(fields_out["W"], W_new)
    assert np.allclose(fields_out["P"], P_new)


def test_bc_hook_called():
    """
    If state["boundary_conditions_post"] is a callable,
    it must be invoked and its returned fields must be used.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    calls = {}

    def bc_post(state, fields):
        calls["called"] = True
        out = dict(fields)
        out["U"] = fields["U"] * 2.0
        return out

    s2["boundary_conditions_post"] = bc_post

    U = np.asarray(s2["fields"]["U"])
    V = np.asarray(s2["fields"]["V"])
    W = np.asarray(s2["fields"]["W"])
    P = np.asarray(s2["fields"]["P"])

    fields_out = apply_boundary_conditions_post(s2, U, V, W, P)

    assert calls.get("called", False)
    assert np.allclose(fields_out["U"], 2.0 * U)


def test_minimal_grid_no_crash():
    """
    Only checks that the function does not crash on a 1×1×1 grid.
    """
    state = {
        "boundary_conditions_post": None,
    }

    U = np.zeros((2, 1, 1))
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))
    P = np.zeros((1, 1, 1))

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape
