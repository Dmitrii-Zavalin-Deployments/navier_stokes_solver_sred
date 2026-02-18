# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_state_update():
    """apply_boundary_conditions_post must return fields identical to inputs."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    U_new = np.ones_like(state.fields["U"])
    V_new = np.ones_like(state.fields["V"])
    W_new = np.ones_like(state.fields["W"])
    P_new = np.ones_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    assert np.allclose(fields_out["U"], U_new)
    assert np.allclose(fields_out["V"], V_new)
    assert np.allclose(fields_out["W"], W_new)
    assert np.allclose(fields_out["P"], P_new)


def test_bc_handler_called():
    """BC handler must be invoked exactly once."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    calls = {"count": 0}

    def bc_handler(state, fields):
        calls["count"] += 1
        out = dict(fields)
        out["U"] = fields["U"] * 2.0
        return out

    state.boundary_conditions = bc_handler

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert calls["count"] == 1
    assert np.allclose(fields_out["U"], 2.0 * U)


def test_solid_mask_zeroing():
    """Velocities adjacent to solid cells must be zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Mark everything as solid
    state.is_fluid[:] = False

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert np.all(fields_out["U"] == 0.0)
    assert np.all(fields_out["V"] == 0.0)
    assert np.all(fields_out["W"] == 0.0)


def test_minimal_grid_no_crash():
    """Function must not crash on minimal grid."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape
